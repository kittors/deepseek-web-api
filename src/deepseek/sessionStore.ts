/**
 * Persists DeepSeek session lineage and resolves continuation requests.
 * Session identity is deliberately model-agnostic so model or thinking changes do not fork.
 */
import fs from "node:fs";
import path from "node:path";
import type { Logger } from "../utils/logger.js";
import { isRecord } from "../utils/json.js";
import { normalizeText } from "../utils/text.js";
import { requestConversationTurns } from "./promptBuild.js";
import { fingerprint, fpKey, turnsEqual, turnsPrefix, turnsSuffix } from "./sessionTurns.js";
import type { MessageTurn, ModelType, RequestBody } from "./types.js";
export type MessageId = string | number | null;
export interface SessionEntry {
  lastResponseMessageId: MessageId;
  lastModelType?: ModelType;
  modelType?: ModelType;
  instructionFingerprint?: string;
  toolsFingerprint?: string;
  updatedAt: number;
  turns: MessageTurn[];
  turnCount?: number;
  totalChars?: number;
}
export interface ConversationResolution {
  sessionId: string | null;
  parentMessageId: MessageId;
  key: string | null;
  pendingFingerprint?: string;
  createIfMissing?: boolean;
  rolloverContext?: MessageTurn[];
}
interface SessionDisk {
  sessions: Record<string, SessionEntry>;
  convs: Record<string, string>;
}
/** Exclude the trailing request turn because it is not stored history yet. */
function historyTurns(messages: unknown): MessageTurn[] {
  if (!Array.isArray(messages)) return [];
  const turns = requestConversationTurns({ messages });
  return turns.at(-1)?.role === "assistant" ? turns : turns.slice(0, -1);
}
function metadata(body: RequestBody): Record<string, unknown> {
  return isRecord(body.metadata) ? body.metadata : {};
}

function stringId(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function parentId(value: unknown): MessageId {
  return typeof value === "string" || typeof value === "number" ? value : null;
}


function parseModelType(value: unknown): ModelType | undefined {
  return value === "default" || value === "expert" ? value : undefined;
}

function parseSessionEntry(value: unknown): SessionEntry | null {
  if (!isRecord(value) || !Array.isArray(value.turns) || typeof value.updatedAt !== "number") return null;
  const turns: MessageTurn[] = [];
  for (const turn of value.turns) {
    if (!isRecord(turn) || typeof turn.role !== "string" || typeof turn.content !== "string") return null;
    turns.push({ role: turn.role, content: turn.content });
  }
  const lastResponseMessageId = parentId(value.lastResponseMessageId);
  const lastModelType = parseModelType(value.lastModelType);
  const modelType = parseModelType(value.modelType);
  const instructionFingerprint = stringId(value.instructionFingerprint) ?? undefined;
  const toolsFingerprint = stringId(value.toolsFingerprint) ?? undefined;
  return {
    lastResponseMessageId,
    updatedAt: value.updatedAt,
    turns,
    ...(typeof value.turnCount === 'number' ? { turnCount: value.turnCount } : {}),
    ...(typeof value.totalChars === 'number' ? { totalChars: value.totalChars } : {}),
    ...(lastModelType ? { lastModelType } : {}),
    ...(modelType ? { modelType } : {}),
    ...(instructionFingerprint ? { instructionFingerprint } : {}),
    ...(toolsFingerprint ? { toolsFingerprint } : {}),
  };
}

const MAX_SESSION_MESSAGES = 80;
const MAX_SESSION_CHARS = 80_000;
const ROLLOVER_CONTEXT_MESSAGES = 12;

/** Public session index used by every completion path. */
export class SessionStore {
  private readonly sessions = new Map<string, SessionEntry>();
  private readonly convIndex = new Map<string, string>();

  constructor(
    private readonly file?: string,
    private readonly logger?: Logger,
  ) {
    this.load();
  }

  has(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }

  get(sessionId: string): SessionEntry | undefined {
    return this.sessions.get(sessionId);
  }

  /** Resolve explicit IDs/history first, then continue the persistent default conversation. */
  resolve(body: RequestBody): ConversationResolution {
    const meta = metadata(body);
    const explicit = stringId(
      body.chat_session_id ??
        body.conversation ??
        body.conversation_id ??
        meta.chat_session_id ??
        meta.conversation_id,
    );
    if (explicit) return this.resolveExplicit(explicit, parentId(body.parent_message_id), "id");

    const previous = stringId(
      body.previous_response_id ?? body.previous_response ?? meta.previous_response_id,
    );
    if (previous) {
      const match = /^resp_([0-9a-f-]{36})/i.exec(previous) ?? /^resp_(.+)$/.exec(previous);
      const sessionId = match?.[1];
      if (sessionId) return this.resolveExplicit(sessionId, parentId(body.parent_message_id), "prev");
    }

    const messages = this.requestMessages(body);
    if (messages && messages.length > 1) {
      const history = this.resolveHistory(messages);
      if (history.sessionId || history.rolloverContext) return history;
      return this.resolveGlobal(history.key, history.pendingFingerprint);
    }
    return this.resolveGlobal(null);
  }

  /** Save the response message ID required as the next parent_message_id. */
  remember(input: {
    sessionId: string;
    modelType: ModelType;
    responseMessageId: MessageId;
    convKey?: string | null;
    prompt: string;
    responseText: string;
    requestTurns?: MessageTurn[];
    instructionFingerprint?: string;
    toolsFingerprint?: string;
  }): void {
    const assistantText = normalizeText(input.responseText);
    if (!assistantText.trim()) {
      this.logger?.warn("Skipping empty assistant turn", {
        sessionId: input.sessionId,
        responseMessageId: input.responseMessageId,
      });
      return;
    }
    const previous = this.sessions.get(input.sessionId);
    const turns = previous ? [...previous.turns] : [];
    const requests = input.requestTurns?.length
      ? input.requestTurns
      : [{ role: "user", content: input.prompt }];
    turns.push(...requests.map((turn) => ({ role: turn.role, content: normalizeText(turn.content) })));
    turns.push({ role: "assistant", content: assistantText });
    const instructionFingerprint = input.instructionFingerprint ?? previous?.instructionFingerprint;
    const toolsFingerprint = input.toolsFingerprint ?? previous?.toolsFingerprint;
    const entry: SessionEntry = {
      lastResponseMessageId: input.responseMessageId ?? previous?.lastResponseMessageId ?? null,
      lastModelType: input.modelType,
      modelType: input.modelType,
      ...(instructionFingerprint ? { instructionFingerprint } : {}),
      ...(toolsFingerprint ? { toolsFingerprint } : {}),
      updatedAt: Date.now(),
      turns: turns.slice(-40),
      turnCount: (previous?.turnCount ?? previous?.turns.length ?? 0) + requests.length + 1,
      totalChars:
        (previous?.totalChars ??
          previous?.turns.reduce((sum, turn) => sum + turn.content.length, 0) ??
          0) +
        requests.reduce((sum, turn) => sum + normalizeText(turn.content).length, 0) +
        assistantText.length,
    };
    this.sessions.set(input.sessionId, entry);
    if (input.convKey) this.convIndex.set(input.convKey, input.sessionId);
    // Index only assistant boundaries so each completed turn can resume independently.
    for (let end = 2; end <= entry.turns.length; end += 2) {
      this.convIndex.set(fpKey(fingerprint(entry.turns.slice(0, end))), input.sessionId);
    }
    this.save();
  }

  private resolveExplicit(
    sessionId: string,
    fallbackParent: MessageId,
    prefix: "id" | "prev",
  ): ConversationResolution {
    const resolvedId =
      prefix === "id"
        ? this.convIndex.get(`id:${sessionId}`) ?? sessionId
        : sessionId;
    const entry = this.sessions.get(resolvedId);

    if (entry && this.needsRollover(entry)) {
      return this.rollover(entry, resolvedId);
    }

    return {
      sessionId: resolvedId,
      parentMessageId: entry?.lastResponseMessageId ?? fallbackParent,
      key: `${prefix}:${sessionId}`,
      ...(!entry ? { createIfMissing: true } : {}),
    };
  }

  private resolveGlobal(
    key: string | null,
    pendingFingerprint?: string,
  ): ConversationResolution {
    // Never attach an unidentified request to another conversation.
    return {
      sessionId: null,
      parentMessageId: null,
      key,
      ...(pendingFingerprint ? { pendingFingerprint } : {}),
    };
  }

  private needsRollover(entry: SessionEntry): boolean {
    const messages = entry.turnCount ?? entry.turns.length;
    const chars = entry.totalChars ?? entry.turns.reduce((sum, turn) => sum + turn.content.length, 0);
    return messages >= MAX_SESSION_MESSAGES || chars >= MAX_SESSION_CHARS;
  }

  private rollover(entry: SessionEntry, oldSessionId: string): ConversationResolution {
    this.logger?.info("Starting a fresh DeepSeek session at the configured safety threshold", {
      previousSessionId: oldSessionId,
      turnCount: entry.turnCount ?? entry.turns.length,
      totalChars: entry.totalChars ?? entry.turns.reduce((sum, turn) => sum + turn.content.length, 0),
    });
    return {
      sessionId: null,
      parentMessageId: null,
      key: null,
      rolloverContext: entry.turns.slice(-ROLLOVER_CONTEXT_MESSAGES),
    };
  }

  private requestMessages(body: RequestBody): unknown[] | null {
    if (Array.isArray(body.messages)) return body.messages;
    if (Array.isArray(body.input) && body.input.some((item) =>
      isRecord(item) && (item.role || ["message", "function_call", "function_call_output"].includes(String(item.type))),
    )) {
      return body.input;
    }
    return null;
  }

  /** Use exact fingerprints before conservative equality, prefix, and assistant-tail matches. */
  private resolveHistory(messages: unknown[]): ConversationResolution {
    const turns = historyTurns(messages);
    if (turns.length === 0) return { sessionId: null, parentMessageId: null, key: null };
    const key = fpKey(fingerprint(turns));
    const exact = this.convIndex.get(key) ?? this.findLegacy(fingerprint(turns));
    if (exact && this.sessions.has(exact)) return this.found(exact, key);

    let best: { sessionId: string; score: number } | undefined;
    for (const [sessionId, entry] of this.sessions) {
      const stored = entry.turns.map((turn) => ({ ...turn, content: normalizeText(turn.content) }));
      let score = 0;
      if (
        turnsEqual(stored, turns) ||
        turnsPrefix(stored, turns) ||
        turnsPrefix(turns, stored) ||
        turnsSuffix(stored, turns) ||
        turnsSuffix(turns, stored)
      ) {
        score = Math.min(stored.length, turns.length);
      } else {
        const lastIncoming = [...turns].reverse().find((turn) => turn.role === "assistant");
        const lastStored = [...stored].reverse().find((turn) => turn.role === "assistant");
        if (
          lastIncoming?.content &&
          normalizeText(lastIncoming.content) === normalizeText(lastStored?.content ?? "")
        ) score = 0.5;
      }
      if (score > 0 && (!best || score > best.score)) best = { sessionId, score };
    }
    if (best) return this.found(best.sessionId, key);
    return { sessionId: null, parentMessageId: null, key, pendingFingerprint: key };
  }

  private found(sessionId: string, key: string): ConversationResolution {
    const entry = this.sessions.get(sessionId);
    if (entry && this.needsRollover(entry)) return this.rollover(entry, sessionId);
    return {
      sessionId,
      parentMessageId: entry?.lastResponseMessageId ?? null,
      key,
    };
  }

  private findLegacy(value: string): string | undefined {
    return ["default", "expert", "vision"]
      .map((model) => this.convIndex.get(`fp:${model}:${value}`))
      .find((sessionId) => sessionId !== undefined);
  }

  private load(): void {
    if (!this.file || !fs.existsSync(this.file)) return;
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(this.file, "utf8"));
      if (!isRecord(parsed) || !isRecord(parsed.sessions) || !isRecord(parsed.convs)) return;
      for (const [id, value] of Object.entries(parsed.sessions)) {
        const entry = parseSessionEntry(value);
        if (entry) this.sessions.set(id, entry);
      }
      for (const [key, value] of Object.entries(parsed.convs)) {
        if (typeof value === "string") this.convIndex.set(key, value);
      }
    } catch (error) {
      this.logger?.warn("无法读取 sessions.json，将使用空会话存储", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private save(): void {
    if (!this.file) return;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true, mode: 0o700 });
      const data: SessionDisk = {
        sessions: Object.fromEntries(this.sessions),
        convs: Object.fromEntries(this.convIndex),
      };
      // Atomic replacement avoids leaving a truncated session file after interruption.
      const temporary = `${this.file}.${process.pid}.tmp`;
      fs.writeFileSync(temporary, JSON.stringify(data, null, 2), { encoding: "utf8", mode: 0o600 });
      fs.renameSync(temporary, this.file);
    } catch (error) {
      this.logger?.error("无法保存 sessions.json", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
}
