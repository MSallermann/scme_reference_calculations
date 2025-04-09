
rule aggregate_json:
    input:
        ["file1.json", "file2.json"],
    output:
        "result.csv",
    params:
        added_columns=dict(column_name=[1, 2]),
    script:
        "scripts/aggregate.py"
