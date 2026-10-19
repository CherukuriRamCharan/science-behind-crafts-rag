"""Robots-aware, rate-limited fetch restricted to configured host suffixes."""
from __future__ import annotations
import time, urllib.parse, urllib.robotparser
import requests, yaml

from .common import resolve_path

class AllowListedFetcher:
    def __init__(self,allowlist_file=None,delay_seconds=1.5,user_agent="SBC-RAG-Prototype/0.1"):
        path=resolve_path(allowlist_file or "allowlist.yaml")
        cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"Allowlist must contain a YAML mapping: {path}")
        self.domains=cfg.get("domains",{})
        self.tier_evidence=cfg.get("tier_evidence",{"official":"E2","scientific":"E3","general":"E4"})
        self.delay=float(delay_seconds); self.user_agent=user_agent; self.last=0.; self.robots={}
    def tier(self,url):
        host=(urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
        for domain,tier in self.domains.items():
            d=domain.lower().lstrip("*.")
            if host==d or host.endswith("."+d): return tier
        return None
    def fetch(self,url,timeout=15):
        tier=self.tier(url)
        if not tier: raise ValueError(f"Host is not allow-listed: {urllib.parse.urlparse(url).hostname}")
        parsed=urllib.parse.urlparse(url); root=f"{parsed.scheme}://{parsed.netloc}"
        if root not in self.robots:
            rp=urllib.robotparser.RobotFileParser(); rp.set_url(root+"/robots.txt")
            try: rp.read()
            except Exception: pass
            self.robots[root]=rp
        if not self.robots[root].can_fetch(self.user_agent,url): raise PermissionError("robots.txt disallows this URL")
        wait=self.delay-(time.monotonic()-self.last)
        if wait>0: time.sleep(wait)
        response=requests.get(url,headers={"User-Agent":self.user_agent},timeout=timeout); self.last=time.monotonic(); response.raise_for_status()
        return {"url":response.url,"status":response.status_code,"tier":tier,"evidence_level":self.tier_evidence.get(tier,"E5"),"content_type":response.headers.get("content-type",""),"text":response.text}

def search(query,provider=None):
    """Search requires a separately configured compliant provider; never scrape SERPs here."""
    if provider is None: return []
    return provider(query)
