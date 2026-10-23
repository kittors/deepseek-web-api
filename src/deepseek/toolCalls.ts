/** Parses the text protocol used to emulate OpenAI function calling. */
import { createHash } from "node:crypto";

import { isRecord } from "../utils/json.js";
import { collectJsonObjects, looksLikeToolJson } from "./toolCallJson.js";

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ParsedToolCalls {
  content: string;
  toolCalls: OpenAIToolCall[];
}

interface Range {
  start: number;
  end: number;
}

interface PayloadCandidate {
  order: number;
  consume: Range;
  payload: { name: string; arguments: Record<string, unknown> };
}

interface TagBlock extends Range {
  bodyStart: number;
  bodyEnd: number;
  attributes: string;
}

const TOOL_TAG = /<\s*(\/?)\s*(tool[_-]?call|_?call)\b([^>]*)>/gi;

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, stableValue(value[key])]),
  );
}

export function stableJson(value: unknown, pretty = false): string {
  return JSON.stringify(stableValue(value), null, pretty ? 2 : undefined) ?? "";
}

function argumentObject(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) {
    if (isRecord(value.arguments) && Object.keys(value).length === 1) {
      return argumentObject(value.arguments);
    }
    return value;
  }
  if (typeof value !== "string") return value === undefined ? {} : null;
  try {
    return argumentObject(JSON.parse(value));
  } catch {
    return null;
  }
}

function callPayload(
  value: unknown,
  attributeName = "",
): { name: string; arguments: Record<string, unknown> } | null {
  if (!isRecord(value)) return null;
  const nested = isRecord(value.function) ? value.function : value;
  const name =
    (typeof nested.name === "string" ? nested.name.trim() : "") ||
    (typeof value.name === "string" ? value.name.trim() : "") ||
    attributeName.trim();
  let argumentsValue = argumentObject(nested.arguments);
  if (!argumentsValue) {
    const rest = Object.fromEntries(
      Object.entries(nested).filter(([key]) => key !== "name" && key !== "function"),
    );
    argumentsValue = Object.keys(rest).length > 0 ? argumentObject(rest) : null;
  }
  return name && argumentsValue ? { name, arguments: argumentsValue } : null;
}

function callId(seed: string, index: number, payload: string): string {
  const digest = createHash("sha256").update(`${seed}:${index}:${payload}`).digest("hex").slice(0, 24);
  return `call_${digest}`;
}

export function formatToolCall(name: string, argumentsValue: unknown): string | null {
  const argumentsObject = argumentObject(argumentsValue);
  if (!name.trim() || !argumentsObject) return null;
  return `<tool_call>\n${stableJson({ name: name.trim(), arguments: argumentsObject })}\n</tool_call>`;
}

function attributeName(attributes: string): string {
  return attributes.match(/\bname\s*=\s*["']([^"']+)["']/i)?.[1]?.trim() ?? "";
}

function tagBlocks(text: string): TagBlock[] {
  TOOL_TAG.lastIndex = 0;
  const tags = [...text.matchAll(TOOL_TAG)].map((match) => ({
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
    closing: Boolean(match[1]),
    attributes: match[3] ?? "",
  }));
  const blocks: TagBlock[] = [];
  for (let index = 0; index < tags.length; index += 1) {
    const open = tags[index];
    if (!open || open.closing) continue;
    const next = tags[index + 1];
    if (next && !next.closing) continue;
    blocks.push({
      start: open.start,
      end: next?.end ?? text.length,
      bodyStart: open.end,
      bodyEnd: next?.start ?? text.length,
      attributes: open.attributes,
    });
    if (next) index += 1;
  }
  return blocks;
}

function overlaps(range: Range, blocked: readonly Range[]): boolean {
  return blocked.some((item) => range.start < item.end && range.end > item.start);
}

function removeRanges(text: string, ranges: readonly Range[]): string {
  const sorted = [...ranges]
    .filter((range) => range.end > range.start)
    .sort((left, right) => left.start - right.start || right.end - left.end);
  let content = "";
  let cursor = 0;
  for (const range of sorted) {
    if (range.end <= cursor) continue;
    content += text.slice(cursor, Math.max(cursor, range.start));
    cursor = range.end;
  }
  return content + text.slice(cursor);
}

function withoutToolTags(text: string): string {
  TOOL_TAG.lastIndex = 0;
  return text.replace(TOOL_TAG, "");
}

function protocolOnly(text: string, ranges: readonly Range[]): boolean {
  return withoutToolTags(removeRanges(text, ranges)).trim().length === 0;
}

function pushUnique(
  toolCalls: OpenAIToolCall[],
  seed: string,
  payload: { name: string; arguments: Record<string, unknown> },
  seen: Set<string>,
): void {
  const argumentsText = stableJson(payload.arguments);
  const key = `${payload.name}\0${argumentsText}`;
  if (seen.has(key)) return;
  seen.add(key);
  toolCalls.push({
    id: callId(seed, toolCalls.length, `${payload.name}:${argumentsText}`),
    type: "function",
    function: { name: payload.name, arguments: argumentsText },
  });
}

function taggedCandidates(text: string, blocks: readonly TagBlock[]): {
  calls: PayloadCandidate[];
  artifacts: Range[];
} {
  const calls: PayloadCandidate[] = [];
  const artifacts: Range[] = [];
  for (const block of blocks) {
    const body = text.slice(block.bodyStart, block.bodyEnd);
    const name = attributeName(block.attributes);
    const objects = collectJsonObjects(body, Boolean(name));
    let valid = false;
    for (const object of objects) {
      const payload = callPayload(object.value, name);
      if (!payload) continue;
      valid = true;
      calls.push({ order: block.bodyStart + object.start, consume: block, payload });
    }
    if (!valid && (looksLikeToolJson(body) || (name && /"arguments"\s*:/.test(body)))) {
      artifacts.push(block);
    }
  }
  return { calls, artifacts };
}


/** Normalize DeepSeek DSML invoke blocks into the existing tool-call protocol. */
function normalizeDsmlToolCalls(text: string): string {
  const invokePattern =
    /<｜｜DSML｜｜\s*invoke\b([^>]*)>([\s\S]*?)(?:<｜｜DSML｜｜\s*\/invoke\s*>|$)/gi;

  return text.replace(invokePattern, (block, attributes: string, body: string) => {
    const name = attributeName(attributes);
    if (!name) return block;

    const args: Record<string, unknown> = {};
    const parameterPattern =
      /<｜｜DSML｜｜\s*parameter\b([^>]*)>([\s\S]*?)<\/｜｜DSML｜｜\s*parameter\s*>/gi;

    let match: RegExpExecArray | null;
    let count = 0;

    while ((match = parameterPattern.exec(body)) !== null) {
      const key = attributeName(match[1] ?? "");
      if (!key) continue;

      args[key] = (match[2] ?? "").replace(/^\r?\n|\r?\n$/g, "");
      count += 1;
    }

    if (count === 0) return block;
    return formatToolCall(name, args) ?? block;
  });
}

/** Parse model tool-call text into OpenAI-compatible calls and cleaned content. */
export function parseToolCalls(text: string, seed = "tool"): ParsedToolCalls {
  text = normalizeDsmlToolCalls(text);
  const blocks = tagBlocks(text);
  const tagged = taggedCandidates(text, blocks);
  const bareObjects = collectJsonObjects(text).filter(
    (object) => !overlaps({ start: object.start, end: object.end }, blocks),
  );
  const bareRanges = bareObjects
    .filter((object) => looksLikeToolJson(object.raw))
    .map((object) => ({ start: object.start, end: object.end }));
  const bareContext = protocolOnly(text, [...blocks, ...bareRanges]);
  const bareCalls: PayloadCandidate[] = bareContext
    ? bareObjects.flatMap((object) => {
        const payload = callPayload(object.value);
        return payload
          ? [{ order: object.start, consume: { start: object.start, end: object.end }, payload }]
          : [];
      })
    : [];
  const candidates = [...tagged.calls, ...bareCalls].sort((left, right) => left.order - right.order);
  const toolCalls: OpenAIToolCall[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) pushUnique(toolCalls, seed, candidate.payload, seen);

  const artifactRanges = [...tagged.artifacts, ...bareRanges];
  const isProtocolOnly = artifactRanges.length > 0 && protocolOnly(text, [...blocks, ...bareRanges]);
  if (toolCalls.length === 0 && !isProtocolOnly) return { content: text, toolCalls };
  const consumed = candidates.map((candidate) => candidate.consume);
  const cleaned = withoutToolTags(
    removeRanges(text, [
      ...consumed,
      ...(toolCalls.length > 0 ? blocks : []),
      ...(bareContext || isProtocolOnly ? artifactRanges : []),
    ]),
  )
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return { content: cleaned, toolCalls };
}

/** Prefer valid output calls, then promote calls leaked into reasoning. */
export function parseToolCallsFromParts(
  outputText: string,
  reasoningText: string,
  seed = "tool",
): ParsedToolCalls {
  const fromOutput = parseToolCalls(outputText, seed);
  if (fromOutput.toolCalls.length > 0) return fromOutput;
  const fromReasoning = parseToolCalls(reasoningText, seed);
  if (fromReasoning.toolCalls.length > 0) {
    return { content: fromOutput.content, toolCalls: fromReasoning.toolCalls };
  }
  return outputText.trim() ? fromOutput : fromReasoning;
}

export function canonicalParsedAssistantText(parsed: ParsedToolCalls): string {
  const calls = parsed.toolCalls
    .map((call) => formatToolCall(call.function.name, call.function.arguments))
    .filter((call): call is string => call !== null);
  return [parsed.content, ...calls].filter(Boolean).join("\n");
}

/** Store a canonical assistant turn so structured client history can match it later. */
export function canonicalAssistantText(text: string): string {
  const parsed = parseToolCalls(text);
  return parsed.toolCalls.length > 0 ? canonicalParsedAssistantText(parsed) : parsed.content;
}
