"""Debug: SearchConstraints — validate the constraints"""

import asyncio

from aweagent.core.tool.search.constraints import SearchConstraints


def test_from_repo():
    print("--- from_repo(owner/repo) ---")
    c = SearchConstraints.from_repo("django/django")
    print(f"repo_owner: {c._repo_owner}")
    print(f"repo_name:  {c._repo_name}")
    for p in c.blocked_patterns["url"]:
        print(f"  pattern: {p}")

    print("\n--- from_repo(repo only) ---")
    c2 = SearchConstraints.from_repo("flask")
    print(f"repo_owner: {c2._repo_owner}")
    print(f"repo_name:  {c2._repo_name}")
    for p in c2.blocked_patterns["url"]:
        print(f"  pattern: {p}")

    print("\n--- from_repo(special chars) ---")
    c3 = SearchConstraints.from_repo("owner/my.repo+v2")
    for p in c3.blocked_patterns["url"]:
        print(f"  pattern: {p}")
    print()


def test_is_url_blocked():
    print("--- is_url_blocked ---")
    c = SearchConstraints.from_repo("django/django")

    cases = [
        ("https://github.com/django/django/pull/42", True),
        ("https://GITHUB.COM/Django/Django/issues", True),
        ("https://gitlab.com/django/django/issues/1", True),
        ("https://github.com/django/django-extensions", False),
        ("https://docs.djangoproject.com/en/5.0/", False),
        ("https://stackoverflow.com/questions/django", False),
        ("https://raw.githubusercontent.com/x/django/main/README", True),
    ]
    for url, expected in cases:
        result = c.is_url_blocked(url)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] {url} -> blocked={result} (expected={expected})")
    print()


def test_filter_search_results():
    print("--- filter_search_results ---")
    c = SearchConstraints.from_repo("django/django")

    results = [
        {"url": "https://github.com/django/django/pull/42", "title": "Fix bug in ORM"},
        {"url": "https://stackoverflow.com/q/123", "title": "Django QuerySet help"},
        {"url": "https://gitlab.com/django/django/issues/1", "title": "Bug report"},
        {"url": "https://docs.djangoproject.com/en/5.0/", "title": "Official docs"},
    ]

    filtered, count = c.filter_search_results(results)
    print(f"  Input: {len(results)} results")
    print(f"  Filtered out: {count}")
    print(f"  Remaining: {len(filtered)}")
    for item in filtered:
        print(f"    - {item['title']} ({item['url']})")
    print()


def test_get_bash_blocklist_patterns():
    print("--- get_bash_blocklist_patterns ---")
    c = SearchConstraints.from_repo("django/django")
    patterns = c.get_bash_blocklist_patterns()
    for p in patterns:
        print(f"  {p}")

    print("\n  Empty constraints -> no patterns:")
    c2 = SearchConstraints()
    print(f"  {c2.get_bash_blocklist_patterns()}")
    print()


def test_merge():
    print("--- merge ---")
    c1 = SearchConstraints.from_repo("django/django")
    c2 = SearchConstraints(blocked_patterns={
        "url": [r".*extra-blocked\.com.*"],
        "title": [r".*CONFIDENTIAL.*"],
    })

    merged = c1.merge(c2)
    print(f"  c1 url patterns: {len(c1.blocked_patterns.get('url', []))}")
    print(f"  c2 url patterns: {len(c2.blocked_patterns.get('url', []))}")
    print(f"  merged url patterns: {len(merged.blocked_patterns.get('url', []))}")
    print(f"  merged title patterns: {merged.blocked_patterns.get('title', [])}")

    # Verify dedup
    c3 = SearchConstraints(blocked_patterns={"url": ["aaa"]})
    c4 = SearchConstraints(blocked_patterns={"url": ["aaa", "bbb"]})
    m = c3.merge(c4)
    print(f"  dedup test: {m.blocked_patterns['url']}")  # should be ['aaa', 'bbb']
    print()


if __name__ == "__main__":
    test_from_repo()
    test_is_url_blocked()
    test_filter_search_results()
    test_get_bash_blocklist_patterns()
    test_merge()
    print("All constraints tests done.")
