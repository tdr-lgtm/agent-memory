from agent_memory.config import get_model


def main() -> None:
    model = get_model()

    reply1 = model.invoke("My name is Dhv.")
    print("Call 1:", reply1.content)

    reply2 = model.invoke("What's my name?")
    print("Call 2:", reply2.content)

    history = [
        ("human", "My name is Dhv."),
        ("ai", reply1.content),
        ("human", "What's my name?"),
    ]
    reply3 = model.invoke(history)
    print("Call 3:", reply3.content)


if __name__ == "__main__":
    main()