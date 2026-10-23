/** Verifies the documented behavior of the corresponding production module. */
import { describe, expect, it } from "vitest";

import { SessionStore } from "../../src/deepseek/sessionStore.js";
import { canonicalAssistantText } from "../../src/deepseek/toolCalls.js";

describe("SessionStore", () => {

  it("does not attach an unidentified new message to the latest session", () => {
    const store = new SessionStore();
    store.remember({
      sessionId: "global-session",
      modelType: "default",
      responseMessageId: 10,
      prompt: "hello",
      responseText: "world",
    });

    expect(store.resolve({ messages: [{ role: "user", content: "continue" }] }))
      .toMatchObject({ sessionId: null, parentMessageId: null });
  });

  it("rolls over a long session and preserves recent context", () => {
    const store = new SessionStore();
    for (let index = 0; index < 40; index += 1) {
      store.remember({
        sessionId: "long-global-session",
        modelType: "default",
        responseMessageId: index + 1,
        prompt: `request-${index}`,
        responseText: `answer-${index}`,
        requestTurns: [{ role: "user", content: `request-${index}` }],
      });
    }

    const result = store.resolve({ conversation: "long-global-session" });
    expect(result.sessionId).toBeNull();
    expect(result.rolloverContext).toHaveLength(12);
    expect(result.rolloverContext?.at(-1)?.content).toBe("answer-39");
  });

  it("does not persist or advance a session for an empty assistant response", () => {
    const store = new SessionStore();
    store.remember({
      sessionId: "empty-session",
      modelType: "expert",
      responseMessageId: 99,
      prompt: "inspect",
      responseText: "   ",
    });

    expect(store.get("empty-session")).toBeUndefined();
    expect(store.resolve({ conversation: "empty-session" })).toMatchObject({
      sessionId: "empty-session",
      parentMessageId: null,
      createIfMissing: true,
    });
  });

  it("matches full history across model switches", () => {
    const store = new SessionStore();
    store.remember({
      sessionId: "session-1",
      modelType: "default",
      responseMessageId: 42,
      prompt: "hello",
      responseText: "world",
    });

    const resolution = store.resolve({
      model: "deepseek-v4-pro",
      messages: [
        { role: "system", content: "Keep the same conversation." },
        { role: "user", content: "hello" },
        { role: "assistant", content: "world" },
        { role: "user", content: "continue" },
      ],
    });

    expect(resolution).toMatchObject({ sessionId: "session-1", parentMessageId: 42 });
  });

  it("reuses a long Pi history after the persisted turn window is truncated", () => {
    const store = new SessionStore();
    const messages: Array<Record<string, unknown>> = [];
    for (let index = 0; index < 22; index += 1) {
      const user = { role: "user", content: `request-${index}` };
      const assistant = { role: "assistant", content: `answer-${index}` };
      messages.push(user, assistant);
      store.remember({
        sessionId: "long-session",
        modelType: "expert",
        responseMessageId: index + 1,
        prompt: `request-${index}`,
        responseText: `answer-${index}`,
        requestTurns: [{ role: "user", content: `request-${index}` }],
      });
    }

    expect(store.get("long-session")?.turns).toHaveLength(40);
    expect(store.resolve({ messages: [...messages, { role: "user", content: "continue" }] }))
      .toMatchObject({ sessionId: "long-session", parentMessageId: 22 });
  });

  it("resolves previous_response_id", () => {
    const store = new SessionStore();
    const sessionId = "123e4567-e89b-12d3-a456-426614174000";
    store.remember({
      sessionId,
      modelType: "expert",
      responseMessageId: 7,
      prompt: "a",
      responseText: "b",
    });
    expect(store.resolve({ previous_response_id: `resp_${sessionId}` })).toMatchObject({
      sessionId,
      parentMessageId: 7,
    });
  });

  it("does not include model type in fingerprint keys", () => {
    const store = new SessionStore();
    store.remember({
      sessionId: "same-session",
      modelType: "default",
      responseMessageId: 1,
      convKey: "fp:user:hello\n---\nassistant:world",
      prompt: "hello",
      responseText: "world",
    });
    expect(
      store.resolve({
        model: "pro",
        input: [
          { role: "user", content: "hello" },
          { role: "assistant", content: "world" },
          { role: "user", content: "next" },
        ],
      }).sessionId,
    ).toBe("same-session");
  });

  it("matches a tool result against the preceding structured assistant call", () => {
    const store = new SessionStore();
    store.remember({
      sessionId: "tool-session",
      modelType: "default",
      responseMessageId: 9,
      prompt: "weather",
      responseText: canonicalAssistantText(
        '<tool_call>\n{"name":"get_weather","arguments":{"city":"Hefei"}}\n</tool_call>',
      ),
    });

    expect(
      store.resolve({
        messages: [
          { role: "user", content: "weather" },
          {
            role: "assistant",
            tool_calls: [{ function: { name: "get_weather", arguments: '{"city":"Hefei"}' } }],
          },
          { role: "tool", content: '{"temperature":32}' },
        ],
      }),
    ).toMatchObject({ sessionId: "tool-session", parentMessageId: 9 });
  });

  it("reuses structured tool history while storing no reasoning prose", () => {
    const store = new SessionStore();
    const canonical = canonicalAssistantText(
      '<tool_call>\n{"name":"bash","arguments":{"command":"date"}}\n</tool_call>',
    );
    store.remember({
      sessionId: "hidden-tool-session",
      modelType: "expert",
      responseMessageId: 12,
      prompt: "今天日期",
      responseText: canonical,
    });

    expect(canonical).toBe(
      '<tool_call>\n{"arguments":{"command":"date"},"name":"bash"}\n</tool_call>',
    );
    expect(canonical).not.toContain("Need current data");
    expect(store.get("hidden-tool-session")?.turns.at(-1)?.content).toBe(
      '<tool_call> {"arguments":{"command":"date"},"name":"bash"} </tool_call>',
    );
    expect(
      store.resolve({
        messages: [
          { role: "user", content: "今天日期" },
          {
            role: "assistant",
            tool_calls: [{ function: { name: "bash", arguments: '{"command":"date"}' } }],
          },
          { role: "tool", tool_call_id: "call_date", content: "Sat Jul 25" },
        ],
      }),
    ).toMatchObject({ sessionId: "hidden-tool-session", parentMessageId: 12 });
  });
});
