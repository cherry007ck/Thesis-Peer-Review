import httpx, feedparser

queries = [
    "Neural Granger Causality causal discovery time series tank",
    "CUTS neural causal discovery time series 2023",
    "causal representation learning nonlinear ICA identifiability",
]

for query in queries:
    params = {"search_query": f"all:{query}", "max_results": 3, "sortBy": "relevance"}
    with httpx.Client(timeout=20, follow_redirects=True) as c:
        r = c.get("https://export.arxiv.org/api/query", params=params)
    feed = feedparser.parse(r.text)
    print(f"\nQuery: {query[:55]}")
    for e in feed.entries[:3]:
        aid = e.get("id", "").split("/abs/")[-1]
        title = e.get("title", "").replace("\n", " ")[:72]
        year = e.get("published", "")[:4]
        print(f"  [{aid}] ({year}) {title}")
