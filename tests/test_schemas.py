import json

from advanced_rag.schemas import ALL_SCHEMAS, RAG_DRILL_DOWN, RAG_LIST_SOURCES, RAG_SEARCH


def test_each_schema_has_required_keys():
    for s in ALL_SCHEMAS:
        assert set(s.keys()) >= {"name", "description", "parameters"}
        assert isinstance(s["name"], str) and s["name"]
        assert isinstance(s["description"], str) and s["description"]
        assert s["parameters"]["type"] == "object"


def test_schemas_serialize_to_json():
    for s in ALL_SCHEMAS:
        # if json.dumps succeeds the schema is at least syntactically valid
        json.dumps(s)


def test_search_requires_query():
    assert "query" in RAG_SEARCH["parameters"]["required"]


def test_drill_down_requires_parent_id():
    assert "parent_id" in RAG_DRILL_DOWN["parameters"]["required"]


def test_list_sources_takes_no_args():
    assert RAG_LIST_SOURCES["parameters"].get("properties", {}) == {}
