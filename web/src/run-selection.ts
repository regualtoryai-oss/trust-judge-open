/** Never truncate a requested batch to the worker limit: require an explicit smaller selection. */
export function selectRunQuestions(availableIds: readonly string[], selection: "all" | readonly string[], limit: number): string[] {
  if (!Number.isInteger(limit) || limit < 1) throw new Error("任务题目上限不可用，请刷新服务状态。");
  const available = new Set(availableIds);
  const ids = [...new Set(selection === "all" ? availableIds : selection)];
  if (!ids.length) throw new Error("当前专题没有可研判的问题。");
  if (ids.some(id => !id || !available.has(id))) throw new Error("所选问题已不在当前专题，请刷新后重试。");
  if (ids.length > limit) throw new Error(`本专题共有 ${ids.length} 个问题，超过单个任务 ${limit} 题上限。请逐题研判，或拆分为不超过 ${limit} 题的专题后分批执行。`);
  return ids;
}
