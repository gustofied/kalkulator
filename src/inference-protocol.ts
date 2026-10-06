export type CompletionReason = "eos" | "limit" | "cancelled";

export type RunTimings = {
  deviceMs?: number;
  tokenizerReadMs?: number;
  tokenizerParseMs?: number;
  artifactMs?: number;
  pipelineMs?: number;
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
  | { readonly type: "cancel"; readonly runId: number };

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
  | {
      readonly type: "done";
      readonly runId: number;
      readonly text: string;
      readonly tokens: number;
      readonly speed: number;
      readonly reason: CompletionReason;
      readonly timings: RunTimings;
    }
  | { readonly type: "rejected"; readonly runId: number; readonly message: string }
  | { readonly type: "error"; readonly message: string; readonly runId?: number };
