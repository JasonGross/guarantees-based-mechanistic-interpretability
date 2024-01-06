def complexity_of(f):
    lines = (line.split(":") for line in f.__doc__.split("\n"))
    lines = (line for line in lines if line[0].lower().strip().startswith("complexity"))
    lines = (
        ":".join(line[1:]).strip()
        if line[0].lower().strip() == "complexity"
        else ":".join(line).strip()[len("complexity") :].strip()
        for line in lines
    )
    return "\n".join(lines)
