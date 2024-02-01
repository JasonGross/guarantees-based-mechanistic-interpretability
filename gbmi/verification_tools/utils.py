from typing import Collection


def complexity_of(
    f, line_starts: Collection[str] = ("complexity", "time complexity")
) -> str:
    lines = (line.split(":") for line in f.__doc__.split("\n"))
    results = []
    for line in lines:
        for start in line_starts:
            if line[0].lower().strip().startswith(start.lower()):
                if line[0].lower().strip() == start.lower():
                    results.append(":".join(line[1:]).strip())
                else:
                    results.append(":".join(line).strip()[len(start) :].strip())
                continue
    return "\n".join(results)
