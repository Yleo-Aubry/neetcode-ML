from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return stoi, itos
            


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        L=[]
        for c in text:
            L.append(stoi[c])
        return L

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        s=''
        for i in ids:
            s+=itos[i]
        return s
