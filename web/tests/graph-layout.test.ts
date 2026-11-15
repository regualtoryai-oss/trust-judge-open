import test from "node:test";
import assert from "node:assert/strict";
import { evidencePositions, focusEvidence } from "../src/graph-layout.ts";
import type { BusinessNode, BusinessEdge } from "../src/types.ts";
const evidence:BusinessNode[]=Array.from({length:17},(_,i)=>({id:`e${i}`,kind:"evidence"}));
const judgments:BusinessNode[]=Array.from({length:5},(_,i)=>({id:`j${i}`,kind:"judgment"}));
const conclusion:BusinessNode={id:"c",kind:"conclusion"};
const nodes=[...evidence,...judgments,conclusion];
const edges:BusinessEdge[]=[{from_node_id:"e16",to_node_id:"j0",relation:"supports"},{from_node_id:"j0",to_node_id:"j1",relation:"prerequisite"}];
test("focus preserves every judgment and conclusion with an explicit evidence limit",()=>{
  const focused=focusEvidence(nodes,edges);
  assert.equal(focused.filter(n=>n.kind==="evidence").length,6);
  assert.equal(focused.filter(n=>n.kind!=="evidence").length,6);
  assert.ok(focused.some(n=>n.id==="e16"));
  assert.equal(nodes.length,23);
});
test("compact focused graph fits readable bounds and respects judgment dependency order",()=>{
  const positions=evidencePositions(focusEvidence(nodes,edges),edges);
  assert.ok(Math.max(...[...positions.values()].map(p=>p.y))<620);
  assert.ok(Math.max(...[...positions.values()].map(p=>p.x))<=750);
  assert.ok(positions.get("j0")!.y<positions.get("j1")!.y);
  assert.notEqual(positions.get("e0")!.x,positions.get("e1")!.x);
});
test("full graph view retains all source cards and no published node is mutated",()=>{
  const before=JSON.stringify(nodes);
  assert.equal(evidencePositions(nodes,edges).size,23);
  assert.equal(JSON.stringify(nodes),before);
  assert.equal(focusEvidence(nodes,edges,100).length,23);
});
