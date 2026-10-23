/** Verifies buffered Chat Completions mapping for tool-compatible turns. */
import { describe, expect, it } from "vitest";

import {
  consumeToolChat,
  emitToolChat,
  streamPlainChat,
  type ChatRun,
  type ChatStreamChunk,
} from "../../src/deepseek/chatStream.js";
import { EMPTY_TOOL_RESPONSE_TEXT } from "../../src/deepseek/toolOutcome.js";

function upstream(parts: { reasoning?: string; output?: string | string[] }): Response {
  const outputs = parts.output
    ? Array.isArray(parts.output) ? parts.output : [parts.output]
    : [];
  const fragments = [
    ...(parts.reasoning ? [{ type: "THINK", content: parts.reasoning }] : []),
    ...outputs.map((content) => ({ type: "RESPONSE", content })),
  ];
  return new Response(
    'event: ready\ndata: {"response_message_id":7}\n\n' +
      `data: {"p":"response/fragments","o":"APPEND","v":${JSON.stringify(fragments)}}\n\n`,
  );
}

function run(stream: Response): ChatRun {
  return { upstream: stream, publicModel: "deepseek-v4-pro", sessionId: "session-1" };
}

function content(chunks: ChatStreamChunk[]): string {
  return chunks.flatMap((chunk) =>
    chunk.choices.flatMap((choice) =>
      typeof choice.delta.content === "string" ? [choice.delta.content] : [],
    ),
  ).join("");
}

describe("plain Chat stream mapping", () => {
  it("streams ordinary text without changing its content", async () => {
    const chunks: ChatStreamChunk[] = [];
    const result = await streamPlainChat(
      run(upstream({ output: ["Hello ", "world."] })),
      (chunk) => chunks.push(chunk),
    );

    expect(content(chunks)).toBe("Hello world.");
    expect(result.finalText).toBe("Hello world.");
  });

  it("suppresses a tool marker split across upstream chunks", async () => {
    const chunks: ChatStreamChunk[] = [];
    const result = await streamPlainChat(
      run(upstream({
        output: [
          "Here is the result.\n<tool_",
          'call>\n{"name":"read","arguments":{"path":"a"}}\n</tool_call>',
        ],
      })),
      (chunk) => chunks.push(chunk),
    );

    expect(content(chunks)).toBe("Here is the result.\n");
    expect(content(chunks)).not.toContain("<tool_call");
    expect(result.finalText).not.toContain("<tool_call");
  });

  it("suppresses a split DSML invoke marker", async () => {
    const chunks: ChatStreamChunk[] = [];
    await streamPlainChat(
      run(upstream({
        output: [
          "Working. <｜｜DSML",
          "｜｜ invoke name=\"shell\"><｜｜parameter name=\"command\">id</｜｜parameter></｜｜DSML｜｜/invoke>",
        ],
      })),
      (chunk) => chunks.push(chunk),
    );

    expect(content(chunks)).toBe("Working. ");
    expect(content(chunks)).not.toContain("DSML");
  });
});

describe("tool Chat stream mapping", () => {
  it("promotes reasoning-only prose into visible content without duplication", async () => {
    const result = await consumeToolChat(run(upstream({ reasoning: "继续读取 src 目录。" })), "");
    const chunks: ChatStreamChunk[] = [];
    emitToolChat(run(new Response()), result, "hidden", (chunk) => chunks.push(chunk));

    expect(content(chunks)).toBe("继续读取 src 目录。");
    expect(JSON.stringify(chunks)).not.toContain("reasoning_content");
    expect(chunks.at(-1)?.choices[0]?.finish_reason).toBe("stop");
  });

  it("emits a visible fallback instead of an empty stop", async () => {
    const result = await consumeToolChat(run(upstream({})), EMPTY_TOOL_RESPONSE_TEXT);
    const chunks: ChatStreamChunk[] = [];
    emitToolChat(run(new Response()), result, "hidden", (chunk) => chunks.push(chunk));

    expect(content(chunks)).toBe(EMPTY_TOOL_RESPONSE_TEXT);
    expect(chunks.at(-1)?.choices[0]?.finish_reason).toBe("stop");
  });

  it("keeps repaired tool calls as tool_calls", async () => {
    const output = '<_call>\n{"name":"read","arguments":{"path":"src/index.ts"}\n</tool_call>';
    const result = await consumeToolChat(run(upstream({ output })), "");
    const chunks: ChatStreamChunk[] = [];
    emitToolChat(run(new Response()), result, "hidden", (chunk) => chunks.push(chunk));

    expect(JSON.stringify(chunks)).toContain('"name":"read"');
    expect(content(chunks)).toBe("");
    expect(chunks.at(-1)?.choices[0]?.finish_reason).toBe("tool_calls");
  });
});
