from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_pilot  # noqa: E402
import run_spider_experiment as spider_run  # noqa: E402
import spider_data  # noqa: E402


class SpiderDataTests(unittest.TestCase):
    def test_schema_prompt_uses_original_names_and_relations(self) -> None:
        table = {
            "table_names_original": ["author", "book"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "name"],
                [1, "id"],
                [1, "author_id"],
            ],
            "column_types": ["text", "number", "text", "number", "number"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]],
        }
        text = spider_data.schema_text(table)
        self.assertIn('"author"("id" number, "name" text)', text)
        self.assertIn('"book"."author_id"="author"."id"', text)
        self.assertNotIn('"*"', text)

    def test_anchor_split_is_database_disjoint_and_deterministic(self) -> None:
        rows = [
            {"db_id": database_id}
            for database_id, count in (("a", 10), ("b", 12), ("c", 18), ("d", 20))
            for _ in range(count)
        ]
        first = spider_data.choose_anchor_database_ids(rows, 21, seed=13)
        second = spider_data.choose_anchor_database_ids(rows, 21, seed=13)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertLess(len(first), 4)

    def test_sql_reward_executes_read_only_and_compares_denotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "toy.sqlite"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE items(id INTEGER, name TEXT)")
            connection.executemany(
                "INSERT INTO items VALUES (?, ?)", [(1, "one"), (2, "two")]
            )
            connection.commit()
            connection.close()
            example = {
                "db_id": "toy",
                "query": "SELECT name FROM items WHERE id = 2",
                "_database_path": str(database),
            }
            correct = spider_data.score_spider(
                "SELECT name FROM items WHERE id = 2;", example
            )
            self.assertTrue(correct["denotation"])
            self.assertEqual(correct["reward"], 1.0)

            mutation = spider_data.score_spider("DROP TABLE items;", example)
            self.assertFalse(mutation["parsed"])
            connection = sqlite3.connect(database)
            self.assertEqual(connection.execute("SELECT count(*) FROM items").fetchone()[0], 2)
            connection.close()

    def test_denotation_comparison_handles_column_permutation(self) -> None:
        gold = [(1, "one"), (2, "two")]
        candidate = [("one", 1), ("two", 2)]
        self.assertTrue(spider_data.denotations_equal(gold, candidate, False))
        self.assertFalse(
            spider_data.denotations_equal(gold, list(reversed(candidate)), True)
        )

    def test_with_query_is_preserved(self) -> None:
        text = "SQL: WITH x AS (SELECT 1 AS n) SELECT n FROM x; explanation"
        self.assertEqual(
            spider_data.extract_spider_sql(text),
            "WITH x AS (SELECT 1 AS n) SELECT n FROM x;",
        )

    def test_official_metric_parser_reads_all_column(self) -> None:
        output = """
        execution 0.1 0.2 0.3 0.4 0.75
        exact match 0.2 0.3 0.4 0.5 0.80
        """
        self.assertEqual(
            spider_run.parse_official_metrics(output),
            {"execution": 0.75, "exact_match": 0.80},
        )

    def test_prompt_ids_are_bounded_from_the_left(self) -> None:
        self.assertEqual(run_pilot.bounded_prompt_ids(range(10), 8, 3), [5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            run_pilot.bounded_prompt_ids([1], 4, 4)


if __name__ == "__main__":
    unittest.main()
