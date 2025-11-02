from fastapi import FastAPI

app = FastAPI()

@app.get("/hello/{name}")
async def helloS(name):
    return f"Welcome ShaonFuck {name}"