import json


def read_jsonl(filepath, limit=None, verbose=False):
    """Read jsonl file to a List of Dicts."""
    data = []
    with open(filepath, "r") as jsonl_file:
      for idx, line in enumerate(jsonl_file):
          if limit is not None and idx >= limit:
              break
          if verbose and idx % 100 == 0:
              # Print the index every 100 lines.
              print("Processing line %s." % idx)
          try:
              data.append(json.loads(line))
          except json.JSONDecodeError as e:
              print("Failed to parse line: `%s`" % line)
              raise e
    print("Loaded %s lines from %s." % (len(data), filepath))
    return data


def write_jsonl(filepath, rows, append=False):
    """Write a List of Dicts to jsonl file."""
    if append:
        jsonl_file = open(filepath, "a")
    else:
        jsonl_file = open(filepath, "w")
    for row in rows:
        line = "%s\n" % json.dumps(row)
        jsonl_file.write(line)
    print("Wrote %s lines to %s." % (len(rows), filepath))
