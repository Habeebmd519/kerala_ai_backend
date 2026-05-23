def build_v3_response(
    answer,
    places=None,
    live_answer=None,
    live_results=None,
    intent=None,
    source="local_data"
):
    return {
        "version": "v3",
        "intent": intent,
        "source": source,
        "answer": answer,
        "places": places or [],
        "liveAnswer": live_answer,
        "liveResults": live_results or []
    }