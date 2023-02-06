import requests

def run(pocket, flop, turn, river):
    if pocket == "":
        pocket = "-"
    if flop == "":
        flop = "-"
    if turn == "":
        turn = "-"
    if river == "":
        river = "-"
    result = requests.get(f"http://localhost:8000/lookup/{pocket}/{flop}/{turn}/{river}").json()
    for key in result:
        blank = " " * (17-len(key))
        print(f"{key} : {blank} {result[key]}")

if __name__ == '__main__':
    pocket = "3d11c"
    flop = "6c12s10h"
    turn = ""
    river = ""
    run(pocket, flop, turn, river)
