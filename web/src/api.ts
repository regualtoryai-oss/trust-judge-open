export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function api<T>(
  path: string,
  body?: unknown,
  idempotent: boolean | string = false,
): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (idempotent) headers["Idempotency-Key"] = typeof idempotent === "string" ? idempotent : crypto.randomUUID();
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "无法连接本地服务，请检查后端是否在 8790 端口运行。");
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.error ?? payload?.message;
    const message =
      typeof detail === "string"
        ? detail
        : typeof detail?.message === "string"
          ? detail.message
          : detail
            ? JSON.stringify(detail)
            : `请求失败（HTTP ${response.status}）`;
    throw new ApiError(response.status, message);
  }
  return payload as T;
}

export function items<T>(value: T[] | { items?: T[] }): T[] {
  return Array.isArray(value) ? value : (value.items ?? []);
}

export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
