import type { BusinessEdge, BusinessNode } from "./types.ts";

const idOf = (node: BusinessNode) => node.id || node.node_id || "";

/** A view filter only: source data, judgment refs and export remain complete. */
export function focusEvidence(nodes: BusinessNode[], edges: BusinessEdge[], limit = 6): BusinessNode[] {
  const evidence = nodes.filter(node => node.kind === "evidence");
  if (evidence.length <= limit) return nodes;
  const degree = new Map<string, number>();
  for (const edge of edges) degree.set(edge.from_node_id, (degree.get(edge.from_node_id) || 0) + 1);
  const chosen = new Set(evidence.map((node,index) => ({ node, index, degree: degree.get(idOf(node)) || 0 }))
    .sort((a,b) => b.degree-a.degree || a.index-b.index).slice(0,limit).map(item => idOf(item.node)));
  return nodes.filter(node => node.kind !== "evidence" || chosen.has(idOf(node)));
}

/** Keep semantic edges intact while arranging compact, readable role columns. */
export function evidencePositions(nodes: BusinessNode[], edges: BusinessEdge[]): Map<string,{x:number;y:number}> {
  const result = new Map<string,{x:number;y:number}>();
  const evidence = nodes.filter(node => node.kind === "evidence");
  const judgments = nodes.filter(node => !["evidence","conclusion"].includes(node.kind));
  const conclusions = nodes.filter(node => node.kind === "conclusion");
  // Real dependency depth orders judgment cards; stable original order breaks ties.
  const depths = new Map(nodes.map(node => [idOf(node),0]));
  for (let pass=0;pass<nodes.length;pass++) {
    let changed=false;
    for (const edge of edges) {
      if (!depths.has(edge.from_node_id) || !depths.has(edge.to_node_id)) continue;
      const next=(depths.get(edge.from_node_id)||0)+1;
      if (next>(depths.get(edge.to_node_id)||0)) { depths.set(edge.to_node_id,next);changed=true; }
    }
    if (!changed) break;
  }
  judgments.sort((a,b) => (depths.get(idOf(a))||0)-(depths.get(idOf(b))||0));
  const evidenceHeight = Math.ceil(evidence.length/2)*108;
  const judgmentHeight = judgments.length*104;
  const conclusionHeight = conclusions.length*140;
  const height = Math.max(evidenceHeight,judgmentHeight,conclusionHeight,300);
  evidence.forEach((node,index) => result.set(idOf(node),{x:20+(index%2)*205,y:35+(height-evidenceHeight)/2+Math.floor(index/2)*108}));
  judgments.forEach((node,index) => result.set(idOf(node),{x:470,y:35+(height-judgmentHeight)/2+index*104}));
  conclusions.forEach((node,index) => result.set(idOf(node),{x:750,y:35+(height-conclusionHeight)/2+index*140}));
  return result;
}
