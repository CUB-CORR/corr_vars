from __future__ import annotations

import pytest

from corr_vars.concepts.routing import (
    ApiRoute,
    ConceptRouting,
    Endpoint,
    LocalRoute,
    load_routing,
    load_routing_document,
    parse_routing,
)
from corr_vars.sources import SOURCES


@pytest.fixture
def routing() -> ConceptRouting:
    return load_routing()


class TestPackagedRoutingFile:
    def test_file_is_importable_from_the_installed_package(self) -> None:
        assert load_routing_document()["defaults"]["taxonomy"] == "corr_v1"

    def test_defaults(self, routing: ConceptRouting) -> None:
        assert routing.taxonomy == "corr_v1"
        assert routing.version == "latest"

    def test_single_wildcard_endpoint(self, routing: ConceptRouting) -> None:
        assert routing.endpoints == (
            Endpoint(url="https://concepts.example.edu/api", sources=("*",)),
        )

    def test_local_datasource_is_pinned_local(self, routing: ConceptRouting) -> None:
        route = routing.resolve("local_datasource")
        assert route == LocalRoute(reason="pinned")
        assert route.is_remote is False

    def test_the_wildcard_endpoint_takes_any_other_source(
        self, routing: ConceptRouting
    ) -> None:
        assert routing.resolve("any_source") == ApiRoute(
            url="https://concepts.example.edu/api"
        )

    def test_reprodicu_routes_to_the_api(self, routing: ConceptRouting) -> None:
        # Routed on purpose even though its definitions are not imported yet.
        assert routing.resolve("reprodicu").is_remote is True

    def test_every_discovered_source_resolves(self, routing: ConceptRouting) -> None:
        resolved = routing.resolve_all(SOURCES)
        assert set(resolved) == set(SOURCES)


class TestResolutionOrder:
    def test_pinned_local_beats_wildcard_endpoint(self) -> None:
        routing = ConceptRouting(
            taxonomy="t",
            version="latest",
            local_sources=("pinned",),
            endpoints=(Endpoint(url="https://a", sources=("*",)),),
        )
        assert routing.resolve("pinned") == LocalRoute(reason="pinned")

    def test_first_matching_endpoint_wins(self) -> None:
        routing = ConceptRouting(
            taxonomy="t",
            version="latest",
            local_sources=(),
            endpoints=(
                Endpoint(url="https://first", sources=("demo_source",)),
                Endpoint(url="https://second", sources=("*",)),
            ),
        )
        assert routing.resolve("demo_source") == ApiRoute("https://first")
        assert routing.resolve("other") == ApiRoute("https://second")

    def test_unmatched_falls_back_to_local(self) -> None:
        routing = ConceptRouting(
            taxonomy="t",
            version="latest",
            local_sources=(),
            endpoints=(Endpoint(url="https://a", sources=("demo_source",)),),
        )
        assert routing.resolve("other") == LocalRoute(reason="unmatched")

    def test_no_endpoints_means_everything_local(self) -> None:
        routing = ConceptRouting("t", "latest", (), ())
        assert routing.resolve("anything").is_remote is False


class TestOverrides:
    def test_taxonomy_and_version(self, routing: ConceptRouting) -> None:
        overridden = routing.with_overrides(taxonomy="corr_v2", version="v3")
        assert overridden.taxonomy == "corr_v2"
        assert overridden.version == "v3"

    def test_api_url_replaces_every_endpoint(self, routing: ConceptRouting) -> None:
        overridden = routing.with_overrides(api_url="https://staging.example/api/")
        assert overridden.resolve("reprodicu") == ApiRoute(
            "https://staging.example/api"
        )

    def test_api_url_does_not_unpin_local_sources(self) -> None:
        routing = ConceptRouting(
            taxonomy="corr_v1",
            version="latest",
            local_sources=("pinned",),
            endpoints=(Endpoint(url="https://a/api", sources=("*",)),),
        )
        overridden = routing.with_overrides(api_url="https://staging.example/api")
        assert overridden.resolve("pinned").is_remote is False

    def test_overrides_do_not_mutate_the_receiver(
        self, routing: ConceptRouting
    ) -> None:
        routing.with_overrides(taxonomy="other", api_url="https://x")
        assert routing.taxonomy == "corr_v1"
        assert routing.endpoints[0].url == "https://concepts.example.edu/api"

    def test_none_keeps_existing_values(self, routing: ConceptRouting) -> None:
        assert routing.with_overrides() == routing


class TestParseRouting:
    def test_trailing_slash_is_stripped(self) -> None:
        routing = parse_routing(
            {
                "defaults": {"taxonomy": "t", "version": "latest"},
                "endpoint": [{"url": "https://a/api/", "sources": ["*"]}],
            }
        )
        assert routing.endpoints[0].url == "https://a/api"

    @pytest.mark.parametrize("key", ["taxonomy", "version"])
    def test_missing_default_is_rejected(self, key: str) -> None:
        defaults = {"taxonomy": "t", "version": "latest"}
        defaults.pop(key)
        with pytest.raises(KeyError, match=key):
            parse_routing({"defaults": defaults})

    def test_string_sources_rejected(self) -> None:
        with pytest.raises(TypeError):
            parse_routing(
                {
                    "defaults": {"taxonomy": "t", "version": "latest"},
                    "local": {"sources": "pinned"},
                }
            )

    def test_endpoint_without_url_rejected(self) -> None:
        with pytest.raises(TypeError):
            parse_routing(
                {
                    "defaults": {"taxonomy": "t", "version": "latest"},
                    "endpoint": [{"sources": ["*"]}],
                }
            )
