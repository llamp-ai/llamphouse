from sqlalchemy import text

def get_max_db_connections(engine):
    with engine.connect() as connection:
        result = connection.execute(text("SHOW max_connections;"))
        max_connections = result.scalar()
        return int(max_connections) if max_connections else None