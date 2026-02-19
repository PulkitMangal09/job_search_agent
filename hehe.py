# Search
from agent.searcher import JobSearcher
searcher = JobSearcher()

results = searcher.search("machine learning pytorch remote", top_k=5)
for job in results:
    print(f"[{job['_score']:.2f}] {job['title']} @ {job['company']} | {job['work_mode']}")