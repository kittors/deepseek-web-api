/** Maps DeepSeek updates to OpenAI-compatible Chat Completions chunks. */
import type { ToolReasoningMode } from "../config/env.js";
import type { CompletionDiagnostics } from "./mapResponses.js";
import type { MessageId } from "./sessionStore.js";
import { canonicalParsedAssistantText, parseToolCalls } from "./toolCalls.js";
import { resolveToolTurn, type ToolTurnOutcome } from "./toolOutcome.js";
import type { PublicModel } from "./types.js";
import { iterDeepSeekUpdates } from "./updates.js";

export interface ChatRun {
  upstream: Response;
  sessionId: string;
  publicModel: PublicModel;
}

export interface ChatStreamChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: PublicModel;
  choices: Array<{
    index: number;
    delta: Record<string, unknown>;
    finish_reason: "stop" | "tool_calls" | null;
  }>;
  conversation?: string;
  previous_response_id?: string;
}

export interface PlainChatResult {
  responseMessageId: MessageId;
  finalText: string;
  diagnostics: CompletionDiagnostics;
}

export interface ToolChatResult {
  outcome: ToolTurnOutcome;
  responseMessageId: MessageId;
}

function chatChunk(
  run: ChatRun,
  created: number,
  delta: Record<string, unknown>,
  finishReason: "stop" | "tool_calls" | null,
): ChatStreamChunk {
  return {
    id: `chatcmpl_${run.sessionId}`,
    object: "chat.completion.chunk",
    created,
    model: run.publicModel,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

function finalChunk(
  run: ChatRun,
  created: number,
  finishReason: "stop" | "tool_calls",
): ChatStreamChunk {
  return {
    ...chatChunk(run, created, {}, finishReason),
    conversation: run.sessionId,
    previous_response_id: `resp_${run.sessionId}`,
  };
}

const PROTOCOL_MARKERS = [
  "<tool_call",
  "<_call",
  "<｜｜DSML｜｜",
];

function findProtocolMarker(text: string): number {
  let first = -1;
  for (const marker of PROTOCOL_MARKERS) {
    const index = text.indexOf(marker);
    if (index >= 0 && (first < 0 || index < first)) first = index;
  }
  return first;
}

function markerPrefixSuffixLength(text: string): number {
  const maxLength = Math.max(...PROTOCOL_MARKERS.map((marker) => marker.length));
  for (let length = Math.min(text.length, maxLength - 1); length > 0; length--) {
    const suffix = text.slice(-length);
    if (PROTOCOL_MARKERS.some((marker) => marker.startsWith(suffix))) return length;
  }
  return 0;
}

export async function streamPlainChat(
  run: ChatRun,
  emit: (chunk: ChatStreamChunk) => void,
): Promise<PlainChatResult> {
  const created = Math.floor(Date.now() / 1000);
  let responseText = "";
  let reasoningText = "";
  let responseMessageId: MessageId = null;
  let roleSent = false;
  let reasoningSent = false;
  let pendingOutput = "";
  let visibleText = "";
  let protocolDetected = false;

  const writeDelta = (delta: Record<string, unknown>): void => {
    const withRole = roleSent ? delta : { role: "assistant", ...delta };
    roleSent = true;
    emit(chatChunk(run, created, withRole, null));
  };

  const writeContent = (text: string): void => {
    if (!text) return;
    if (!reasoningSent && reasoningText) {
      writeDelta({ reasoning_content: reasoningText });
      reasoningSent = true;
    }
    writeDelta({ content: text });
    visibleText += text;
  };

  for await (const update of iterDeepSeekUpdates(run.upstream)) {
    if (update.type === "ready") {
      responseMessageId = update.responseMessageId ?? responseMessageId;
    } else if (update.type === "reasoning" && update.delta) {
      reasoningText += update.delta;
      if (visibleText && reasoningSent) {
        writeDelta({ reasoning_content: update.delta });
      }
    } else if (update.type === "output" && update.delta) {
      responseText += update.delta;
      if (protocolDetected) continue;

      pendingOutput += update.delta;
      const markerIndex = findProtocolMarker(pendingOutput);

      if (markerIndex >= 0) {
        writeContent(pendingOutput.slice(0, markerIndex));
        protocolDetected = true;
        pendingOutput = "";
        continue;
      }

      // Retain only a possible marker prefix so split markers cannot leak.
      const retained = markerPrefixSuffixLength(pendingOutput);
      const safeLength = pendingOutput.length - retained;
      writeContent(pendingOutput.slice(0, safeLength));
      pendingOutput = pendingOutput.slice(safeLength);
    }
  }

  if (!protocolDetected && pendingOutput &&
      !PROTOCOL_MARKERS.some((marker) => marker.startsWith(pendingOutput))) {
    writeContent(pendingOutput);
  } else if (!protocolDetected && pendingOutput) {
    // Do not expose a truncated protocol marker at end of stream.
    protocolDetected = true;
  }

  let finalText: string;
  if (protocolDetected) {
    const parsed = parseToolCalls(responseText);
    // A plain-chat request must never expose protocol syntax as assistant prose.
    if (parsed.toolCalls.length > 0 && parsed.content.startsWith(visibleText)) {
      writeContent(parsed.content.slice(visibleText.length));
      finalText = parsed.content;
    } else {
      finalText = visibleText;
    }
  } else {
    finalText = responseText.trim() ? responseText : reasoningText;
    if (!responseText.trim() && reasoningText) writeContent(reasoningText);
  }

  emit(finalChunk(run, created, "stop"));
  return {
    responseMessageId,
    finalText,
    diagnostics: {
      reasoningChars: reasoningText.length,
      outputChars: responseText.length,
      toolCallCount: 0,
      emptyUpstream: !reasoningText.trim() && !responseText.trim(),
      recoverableEmpty: false,
      promotedReasoning: !responseText.trim() && Boolean(reasoningText.trim()),
    },
  };
}

export async function consumeToolChat(run: ChatRun, emptyFallback: string): Promise<ToolChatResult> {
  let responseText = "";
  let reasoningText = "";
  let responseMessageId: MessageId = null;
  for await (const update of iterDeepSeekUpdates(run.upstream)) {
    if (update.type === "ready") {
      responseMessageId = update.responseMessageId ?? responseMessageId;
    } else if (update.type === "reasoning" && update.delta) reasoningText += update.delta;
    else if (update.type === "output" && update.delta) responseText += update.delta;
  }
  return {
    outcome: resolveToolTurn(responseText, reasoningText, `chatcmpl_${run.sessionId}`, emptyFallback),
    responseMessageId,
  };
}

export function toolDiagnostics(outcome: ToolTurnOutcome): CompletionDiagnostics {
  return {
    reasoningChars: outcome.reasoningText.length,
    outputChars: outcome.outputText.length,
    toolCallCount: outcome.parsed.toolCalls.length,
    emptyUpstream: outcome.emptyUpstream,
    recoverableEmpty: outcome.recoverableEmpty,
    promotedReasoning: outcome.promotedReasoning,
  };
}

/** Emit one buffered tool-compatible result after retry selection. */
export function emitToolChat(
  run: ChatRun,
  result: ToolChatResult,
  toolReasoning: ToolReasoningMode,
  emit: (chunk: ChatStreamChunk) => void,
): string {
  const created = Math.floor(Date.now() / 1000);
  const parsed = result.outcome.parsed;
  let roleSent = false;
  const writeDelta = (delta: Record<string, unknown>): void => {
    const withRole = roleSent ? delta : { role: "assistant", ...delta };
    roleSent = true;
    emit(chatChunk(run, created, withRole, null));
  };
  const finishReason = parsed.toolCalls.length > 0 ? "tool_calls" : "stop";

  if (parsed.toolCalls.length > 0) {
    if (toolReasoning === "clean") {
      const cleanedReason = parseToolCalls(result.outcome.reasoningText).content;
      if (cleanedReason) writeDelta({ reasoning_content: cleanedReason });
    }
    if (parsed.content) writeDelta({ content: parsed.content });
    writeDelta({ tool_calls: parsed.toolCalls.map((call, index) => ({ index, ...call })) });
  } else {
    if (result.outcome.reasoningText && result.outcome.outputText.trim()) {
      writeDelta({ reasoning_content: result.outcome.reasoningText });
    }
    writeDelta({ content: parsed.content });
  }

  emit(finalChunk(run, created, finishReason));
  return parsed.toolCalls.length > 0 ? canonicalParsedAssistantText(parsed) : parsed.content;
}
