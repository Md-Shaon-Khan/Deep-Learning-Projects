from fastapi import FastAPI
from enum import Enum


app = FastAPI()

class AvailableCuisines(str,Enum):
    bangladeshi = "bangladeshi"
    american = "american"
    nepali = "nepali"
    
food_item = {
    'bangladeshi' : ['Biriyani','Pitha'],
    'american' : ['Hot Dog','Burger'],
    'nepali' : ['Samosa','Dosa']
    
}

valid_cuisines = food_item.keys()
@app.get("/get_items/{cuisine}")
async def get_items(cuisine: AvailableCuisines):
    return {"cuisine": cuisine, "items": food_item[cuisine]}
