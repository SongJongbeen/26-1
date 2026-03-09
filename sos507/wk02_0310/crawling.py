import requests
import json
import time

def get_openalex_source_id(issn: str) -> str | None:
    """ISSN으로 OpenAlex Source ID를 먼저 조회"""
    resp = requests.get(
        "https://api.openalex.org/sources",
        params={"filter": f"issn:{issn}"},
        headers={"User-Agent": "MyResearchBot/1.0 (mailto:your@email.com)"},
        timeout=30
    ).json()
    results = resp.get("results", [])
    if results:
        source_id = results[0]["id"]  # e.g. "https://openalex.org/S1234567890"
        count     = results[0].get("works_count", "?")
        print(f"  Source ID: {source_id}")
        print(f"  전체 works count: {count}")
        return source_id
    return None


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    if not inverted_index:
        return None
    positions = []
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def get_papers_by_source_id(
    source_id: str,
    year_from: int = 2010,
    year_to:   int = 2026,
) -> list[dict]:
    BASE_URL = "https://api.openalex.org/works"
    headers  = {"User-Agent": "MyResearchBot/1.0 (mailto:your@email.com)"}
    results  = []
    cursor   = "*"

    while cursor:
        params = {
            # Source ID + 연도 범위로 이중 필터
            "filter":   (
                f"primary_location.source.id:{source_id},"
                f"publication_year:{year_from}-{year_to}"
            ),
            "per-page": 100,
            "cursor":   cursor,
            "select":   "title,abstract_inverted_index,doi,publication_year",
        }

        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30).json()
        items = resp.get("results", [])
        if not items:
            break

        for item in items:
            results.append({
                "title":            item.get("title"),
                "abstract":         reconstruct_abstract(item.get("abstract_inverted_index")),
                "doi":              item.get("doi"),
                "publication_year": item.get("publication_year"),
            })

        cursor = resp.get("meta", {}).get("next_cursor")
        print(f"  수집 중... {len(results)}건")
        time.sleep(0.2)

    return results


if __name__ == "__main__":
    ISSN = "1097-3729"

    print("🔍 OpenAlex Source ID 조회 중...")
    source_id = get_openalex_source_id(ISSN)

    if not source_id:
        print("❌ Source ID를 찾을 수 없습니다.")
        exit()

    print(f"\n📄 논문 수집 시작 (Source ID 기반, 연도 필터 적용)...")
    papers = get_papers_by_source_id(source_id, year_from=2010, year_to=2026)

    with_abstract = [p for p in papers if p["abstract"]]
    print(f"\n✅ 총 {len(papers)}편 | 초록 보유 {len(with_abstract)}편")

    with open("full_abstracts.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    with open("with_abstracts.json", "w", encoding="utf-8") as f:
        json.dump(with_abstract, f, ensure_ascii=False, indent=2)

    print("💾 full_abstracts.json 저장 완료")
    print("💾 with_abstracts.json 저장 완료")
