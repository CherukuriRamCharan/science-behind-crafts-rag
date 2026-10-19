"""Retrieval metrics and lightweight, deterministic groundedness proxies."""
from __future__ import annotations
import math
from .common import read_json, write_json
from .generate import GroundedGenerator
from .retrieve import HybridRetriever

def retrieval_metrics(retrieved_ids,relevant_ids,k):
    rel=set(relevant_ids); top=retrieved_ids[:k]
    precision=sum(x in rel for x in top)/max(k,1); recall=sum(x in rel for x in top)/max(len(rel),1)
    rr=next((1/(i+1) for i,x in enumerate(retrieved_ids) if x in rel),0.)
    dcg=sum((1/math.log2(i+2)) for i,x in enumerate(top) if x in rel)
    idcg=sum(1/math.log2(i+2) for i in range(min(len(rel),k)))
    return {"precision_at_k":precision,"recall_at_k":recall,"mrr":rr,"ndcg_at_k":dcg/idcg if idcg else 0.}

def evaluate(config,index,out_dir=None,generator=None,evalset=None,eval_source=None):
    evalset=evalset if evalset is not None else read_json(config["data"]["eval_file"])
    retriever=HybridRetriever(index,config); generator=generator or GroundedGenerator(config)
    k=config["retrieval"].get("top_k",5); rows=[]; sums={x:0. for x in ("precision_at_k","recall_at_k","mrr","ndcg_at_k")}
    faith=[]; relevancy=[]
    for item in evalset:
        found=retriever.retrieve(item["query"],k); ids=list(dict.fromkeys(p["source_id"] for p in found))
        metrics=retrieval_metrics(ids,item["relevant_source_ids"],k)
        for name,value in metrics.items(): sums[name]+=value
        answer=generator.answer(item["query"],found)
        cited=any(f"[{p['source_id']} | {p.get('evidence_level','E5')}]" in answer for p in found)
        overlap=bool(found and set(item["query"].lower().split()) & set(" ".join(p["text_chunk"].lower() for p in found).split()))
        faith.append(float(cited and (overlap or answer.startswith("Insufficient evidence"))))
        relevancy.append(float(bool(found) and not answer.startswith("Insufficient evidence")))
        rows.append({"query":item["query"],"expected":item["relevant_source_ids"],"retrieved":ids,"metrics":metrics,"answer":answer,"citation_proxy":cited})
    n=max(len(evalset),1); report={"queries":len(evalset),"evaluation_source":eval_source or config["data"].get("eval_file"),"retrieval":{k:v/n for k,v in sums.items()},"generation_proxy":{"faithfulness":sum(faith)/n,"answer_relevancy":sum(relevancy)/n},"details":rows}
    root=out_dir or config["data"].get("artifact_dir","artifacts"); write_json(f"{root}/evaluation.json",report)
    md=["| Metric | Score |","|---|---:|"]+ [f"| {name} | {val:.3f} |" for name,val in report["retrieval"].items()]+[f"| groundedness proxy | {report['generation_proxy']['faithfulness']:.3f} |",f"| answer relevancy proxy | {report['generation_proxy']['answer_relevancy']:.3f} |"]
    from pathlib import Path
    Path(root).mkdir(parents=True,exist_ok=True); Path(root,"evaluation.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    return report

def format_table(report):
    r=report["retrieval"]; g=report["generation_proxy"]
    lines=["Metric                     Score","-------------------------  -----"]
    lines += [f"Precision@k                 {r['precision_at_k']:.3f}",f"Recall@k                    {r['recall_at_k']:.3f}",f"MRR                         {r['mrr']:.3f}",f"nDCG@k                      {r['ndcg_at_k']:.3f}",f"Faithfulness proxy          {g['faithfulness']:.3f}",f"Answer relevancy proxy      {g['answer_relevancy']:.3f}"]
    return "\n".join(lines)
