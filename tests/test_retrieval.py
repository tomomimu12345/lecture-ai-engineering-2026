from app import retrieval


def _page(number: int, section: str = "Part 7", document_id: str = "lec03-main") -> dict:
    return {
        "page_id": f"{document_id}-p{number:03}",
        "document_id": document_id,
        "lecture": 3,
        "document_title": "AIエージェント",
        "section": section,
        "title": f"ページ{number}",
        "page": number,
        "content": f"本文{number}",
    }


def test_section_graph_adds_nearby_pages_then_deduplicates_graph_results(monkeypatch):
    pages = [_page(number) for number in range(42, 49)]
    pages.extend([_page(49, section="Part 9"), _page(50, section="Part 9")])
    seed = {**pages[6], "distance": 0.1, "hops": 0, "path": []}
    graph_neighbor = {**pages[5], "hops": 1, "path": []}
    other_graph_page = {**pages[7], "hops": 1, "path": []}

    def fake_search_graph(query: str) -> dict:
        assert query == "autoresearch と冪等性"
        return {
            "seeds": [seed],
            "expanded": [graph_neighbor, other_graph_page],
            "max_hops": 2,
        }

    def fake_load_pages(source) -> list[dict]:
        assert source == retrieval.RAG_SOURCE
        return pages

    monkeypatch.setattr(
        retrieval,
        "search_graph",
        fake_search_graph,
    )
    monkeypatch.setattr(retrieval, "load_pages", fake_load_pages)
    monkeypatch.setattr(retrieval, "GRAPH_MAX_RESULTS", 8)

    result = retrieval.search_graph_with_section_context("autoresearch と冪等性")

    ids = [page["page_id"] for page in result["seeds"] + result["section_expanded"] + result["expanded"]]
    assert ids == [
        "lec03-main-p048",
        "lec03-main-p047",
        "lec03-main-p046",
        "lec03-main-p045",
        "lec03-main-p044",
        "lec03-main-p049",
    ]
    assert len(ids) == len(set(ids)) <= 8
    assert all(page["anchor_page_id"] == seed["page_id"] for page in result["section_expanded"])
