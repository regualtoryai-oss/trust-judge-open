<script setup lang="ts">
import { Handle, Position } from "@vue-flow/core";
import type { NodeProps } from "@vue-flow/core";
import type { BusinessNode } from "../types";
import { nodeText } from "../presentation";
defineProps<NodeProps<BusinessNode>>();
const kinds: Record<string, string> = {
  evidence: "原文证据",
  judgment: "中间判断",
  intermediate: "中间判断",
  conclusion: "最终结论",
};
</script>

<template>
  <article class="judgment-node node-arrive" :class="`node-${data.kind}`">
    <Handle id="left" type="target" :position="Position.Left" />
    <Handle id="top" type="target" :position="Position.Top" />
    <div class="node-topline">
      <span>{{ kinds[data.kind] ?? data.kind }}</span
      ><span v-if="data.locked">人工锁定</span>
    </div>
    <p :title="nodeText(data)">{{ nodeText(data) }}</p>
    <small>{{ data.status || "未标记状态" }}</small>
    <Handle id="right" type="source" :position="Position.Right" />
    <Handle id="bottom" type="source" :position="Position.Bottom" />
  </article>
</template>
