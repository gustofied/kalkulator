export type CompletionReason = "eos" | "limit" | "cancelled" | "timeout" | "repetition";

export type RunTimings = {
  deviceMs?: number;
  tokenizerReadMs?: number;
  tokenizerParseMs?: number;
  artifactMs?: number;
  prefillMs?: number;
  firstTokenMs?: number;
  totalMs?: number;
};

export type InferenceRequest =
  | { readonly type: "prepare" }
  | {
      readonly type: "solve";
      readonly runId: number;
      readonly problem: string;
    }
  | { readonly type: "continue"; readonly runId: number }
  | { readonly type: "cancel"; readonly runId: number };

type GenerationReport = {
  readonly runId: number;
  readonly text: string;
  readonly tokens: number;
  readonly speed: number;
  readonly reason: CompletionReason;
  readonly timings: RunTimings;
};

export type InferenceResponse =
  | {
      readonly type: "status";
      readonly status: string;
      readonly progress?: number;
      readonly runId?: number;
    }
  | { readonly type: "ready"; readonly timings: RunTimings }
  | {
      readonly type: "update";
      readonly runId: number;
      readonly delta: string;
      readonly tokens: number;
      readonly speed: number;
    }
  | ({ readonly type: "done" } & GenerationReport)
  | ({ readonly type: "paused" } & GenerationReport)
  | { readonly type: "rejected"; readonly runId: number; readonly message: string }
  | {
      readonly type: "error";
      readonly message: string;
      readonly runId?: number;
      readonly reloadRequired?: boolean;
    };
