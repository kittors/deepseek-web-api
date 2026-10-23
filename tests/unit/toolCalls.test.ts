/** Verifies parsing of the text protocol into OpenAI-compatible tool calls. */
import { describe, expect, it } from "vitest";

import {
  canonicalAssistantText,
  canonicalParsedAssistantText,
  parseToolCalls,
  parseToolCallsFromParts,
} from "../../src/deepseek/toolCalls.js";

const FAILED_SESSION_A = `{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/.github/workflows/ci.yml"}

{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/tsconfig.json"}`;

const FAILED_SESSION_B = `<_call>
{"name": "bash", "arguments": {"command": "find /Users/kittors/Developer/opensource/deepseek-web-api/src -name '*.ts' | sort", "timeout": 5}
</tool_call>`;

const FAILED_SESSION_C = `<_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/client.ts"}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/server/routes.ts"}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/completion.ts"}
</tool_call>`;


describe("DeepSeek DSML tool calls", () => {
  it("parses DSML invoke calls with a file path and content", () => {
    const text = `<｜｜DSML｜｜ invoke name="write">
<｜｜DSML｜｜parameter name="filePath">/home/ubuntu/test.txt</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="content">Hello from opencode!</｜｜DSML｜｜parameter>
<｜｜DSML｜｜/invoke>`;
    const result = parseToolCalls(text, "dsml-write");

    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("write");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      filePath: "/home/ubuntu/test.txt",
      content: "Hello from opencode!",
    });
    expect(result.content).toBe("");
  });

  it("parses DSML shell calls", () => {
    const text = `<｜｜DSML｜｜ invoke name="shell">
<｜｜DSML｜｜parameter name="command">printf 'TOOL_OK\\\\n'; whoami</｜｜DSML｜｜parameter>
<｜｜DSML｜｜/invoke>`;
    const result = parseToolCalls(text, "dsml-shell");

    expect(result.toolCalls[0]?.function.name).toBe("shell");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      command: "printf 'TOOL_OK\\\\n'; whoami",
    });
  });
});

describe("parseToolCalls", () => {
  it("maps tagged JSON to OpenAI tool_calls while preserving normal content", () => {
    const result = parseToolCalls(
      'Checking first.\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Hefei"}}\n</tool_call>',
      "response-1",
    );

    expect(result.content).toBe("Checking first.");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]).toMatchObject({
      id: expect.stringMatching(/^call_[0-9a-f]{24}$/),
      type: "function",
      function: { name: "get_weather", arguments: '{"city":"Hefei"}' },
    });
  });

  it("leaves malformed call text as ordinary content", () => {
    const text = "<tool_call>{not json}</tool_call>";
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });

  it("leaves pure prose unchanged", () => {
    const text = 'Use JSON like {"name":"example"} when documenting payloads.';
    expect(parseToolCalls(text)).toEqual({ content: text, toolCalls: [] });
  });

  it("accepts attribute-name tags used by DeepSeek/Pi hybrid output", () => {
    const sample = `<_call>
<tool_call name="read">
{"arguments": {"path": "/tmp/README.md", "limit": 100}}
</tool_call>
<tool_call name="read">
{"arguments": {"path": "/tmp/SPEC.md", "limit": 100}}
</tool_call>
<tool_call name="read">
{"arguments": {"arguments": {"path": "/tmp/package.json"}}}
</tool_call>
<tool_call name="bash">
{"arguments": {"command": "find src -type f | head -5", "timeout": 10}}
</tool_call>
</tool_call>`;

    const result = parseToolCalls(sample, "seed");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual([
      "read",
      "read",
      "read",
      "bash",
    ]);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      path: "/tmp/README.md",
      limit: 100,
    });
    expect(JSON.parse(result.toolCalls[2]?.function.arguments ?? "{}")).toEqual({
      path: "/tmp/package.json",
    });
    expect(JSON.parse(result.toolCalls[3]?.function.arguments ?? "{}")).toEqual({
      command: "find src -type f | head -5",
      timeout: 10,
    });
    expect(result.content).toBe("");
  });

  it("parses name-first bare JSON and preserves canonical arguments-name key order", () => {
    const bare = '{"name":"bash","arguments":{"timeout":10,"command":"pwd"}}';
    const result = parseToolCalls(bare, "seed");

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "bash",
      arguments: '{"command":"pwd","timeout":10}',
    });
    expect(canonicalAssistantText(bare)).toBe(
      '<tool_call>\n{"arguments":{"command":"pwd","timeout":10},"name":"bash"}\n</tool_call>',
    );
  });

  it("accepts mangled tags and bare JSON tool payloads", () => {
    const mangled = parseToolCalls(
      '<_call>\n{"name":"list_files","arguments":{"path":"."}}\n</tool_call>',
      "seed",
    );
    expect(mangled.toolCalls).toHaveLength(1);
    expect(mangled.toolCalls[0]?.function).toEqual({
      name: "list_files",
      arguments: '{"path":"."}',
    });

    const bare = parseToolCalls('{"name":"read_file","arguments":{"path":"README.md"}}', "seed");
    expect(bare.toolCalls[0]?.function.name).toBe("read_file");
  });

  it("harvests args-before-name bare JSON alongside tagged calls", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"read","arguments":{"path":"README.md"}}
</tool_call>
{"arguments":{"path":"package.json"},"name":"read"}`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function)).toEqual([
      { name: "read", arguments: '{"path":"README.md"}' },
      { name: "read", arguments: '{"path":"package.json"}' },
    ]);
  });

  it("deduplicates identical tagged and bare calls while stripping both", () => {
    const payload = '{"arguments":{"path":"package.json"},"name":"read"}';
    const result = parseToolCalls(`<tool_call>${payload}</tool_call>\n${payload}`);

    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"package.json"}',
    });
  });

  it("repairs both truncated bare reads from failed session A", () => {
    const result = parseToolCalls(FAILED_SESSION_A, "session-a");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["read", "read"]);
    expect(result.toolCalls.map((call) => JSON.parse(call.function.arguments))).toEqual([
      {
        path: "/Users/kittors/Developer/opensource/deepseek-web-api/.github/workflows/ci.yml",
      },
      { path: "/Users/kittors/Developer/opensource/deepseek-web-api/tsconfig.json" },
    ]);
  });

  it("repairs the cross-closed truncated bash from failed session B", () => {
    const result = parseToolCalls(FAILED_SESSION_B, "session-b");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["bash"]);
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      command: "find /Users/kittors/Developer/opensource/deepseek-web-api/src -name '*.ts' | sort",
      timeout: 5,
    });
    expect(canonicalParsedAssistantText(result)).toBe(
      '<tool_call>\n{"arguments":{"command":"find /Users/kittors/Developer/opensource/deepseek-web-api/src -name \'*.ts\' | sort","timeout":5},"name":"bash"}\n</tool_call>',
    );
  });

  it.each([
    ["<_call>", "</tool_call>"],
    ["<_call>", "</_call>"],
    ["<tool_call>", "</_call>"],
    ["<tool_call>", "</tool_call>"],
  ])("repairs loose tag pair %s ... %s", (open, close) => {
    const result = parseToolCalls(
      `${open}\n{"name":"read","arguments":{"path":"README.md"}\n${close}`,
    );

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"README.md"}',
    });
  });

  it("repairs all three truncated reads from failed session C", () => {
    const result = parseToolCalls(FAILED_SESSION_C, "session-c");

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["read", "read", "read"]);
    expect(result.toolCalls.map((call) => JSON.parse(call.function.arguments).path)).toEqual([
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/client.ts",
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/server/routes.ts",
      "/Users/kittors/Developer/opensource/deepseek-web-api/src/deepseek/completion.ts",
    ]);
  });

  it("repairs several missing nested closers without changing their data", () => {
    const result = parseToolCalls(
      '{"name":"write","arguments":{"config":{"items":[{"path":"a"},{"path":"b"}]',
    );

    expect(result.content).toBe("");
    expect(JSON.parse(result.toolCalls[0]?.function.arguments ?? "{}")).toEqual({
      config: { items: [{ path: "a" }, { path: "b" }] },
    });
  });

  it("harvests a complete tagged call beside a truncated bare call", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"bash","arguments":{"command":"pwd"}}
</tool_call>
{"arguments":{"path":"package.json"},"name":"read"`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function)).toEqual([
      { name: "bash", arguments: '{"command":"pwd"}' },
      { name: "read", arguments: '{"path":"package.json"}' },
    ]);
  });

  it("keeps a valid bash call while repairing both failed session A reads", () => {
    const result = parseToolCalls(`<tool_call>
{"name":"bash","arguments":{"command":"ls -la /Users/kittors/Developer/opensource/deepseek-web-api/src/","timeout":5}}
</tool_call>
${FAILED_SESSION_A}`);

    expect(result.content).toBe("");
    expect(result.toolCalls.map((call) => call.function.name)).toEqual(["bash", "read", "read"]);
  });

  it("repairs an unclosed tagged block at end of output", () => {
    const result = parseToolCalls('<tool_call>\n{"name":"read","arguments":{"path":"README.md"');

    expect(result.content).toBe("");
    expect(result.toolCalls[0]?.function).toEqual({
      name: "read",
      arguments: '{"path":"README.md"}',
    });
  });

  it("does not swallow prose or fenced JSON documentation examples", () => {
    const prose = 'Example payload:\n{"name":"read","arguments":{"path":"README.md"}}';
    const fenced = '```json\n{"name":"read","arguments":{"path":"README.md"}}\n```';

    expect(parseToolCalls(prose)).toEqual({ content: prose, toolCalls: [] });
    expect(parseToolCalls(fenced)).toEqual({ content: fenced, toolCalls: [] });
  });

  it("cleans protocol-only garbage when an unfinished string cannot be repaired safely", () => {
    const garbage = '<_call>\n{"name":"read","arguments":{"path":"README.md}\n</tool_call>';

    expect(parseToolCalls(garbage)).toEqual({ content: "", toolCalls: [] });
  });

  it("promotes tool tags leaked into reasoning when output has none", () => {
    const result = parseToolCallsFromParts(
      "",
      'I should list files.\n<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>',
      "seed",
    );
    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function).toEqual({
      name: "bash",
      arguments: '{"command":"ls"}',
    });
    expect(canonicalParsedAssistantText(result)).not.toContain("I should list files.");
  });

  it("prefers output tool calls over reasoning ones", () => {
    const result = parseToolCallsFromParts(
      '<tool_call>\n{"name":"read","arguments":{"path":"a"}}\n</tool_call>',
      '<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n</tool_call>',
      "seed",
    );
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("read");
  });

  it("promotes a truncated call found only in reasoning", () => {
    const result = parseToolCallsFromParts("", FAILED_SESSION_B, "reasoning-repair");

    expect(result.content).toBe("");
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.function.name).toBe("bash");
  });

  it("returns cleaned reasoning when no RESPONSE text exists", () => {
    expect(parseToolCallsFromParts("", "A normal final answer.")).toEqual({
      content: "A normal final answer.",
      toolCalls: [],
    });
    expect(
      parseToolCallsFromParts(
        "",
        '<_call>\n{"name":"read","arguments":{"path":"README.md}\n</tool_call>',
      ),
    ).toEqual({ content: "", toolCalls: [] });
  });
});
