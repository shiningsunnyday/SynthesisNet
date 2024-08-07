import uuid
from pathlib import Path
from typing import Optional, Union

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Draw, rdChemReactions


class MolDrawer:
    """Draws molecules as images."""

    def __init__(self, path: Optional[str], subfolder: str = "assets"):

        # Init outfolder
        if not (path is not None and Path(path).exists()):
            raise NotADirectoryError(path)
        self.outfolder = Path(path) / subfolder
        self.outfolder.mkdir(exist_ok=1)

        # Placeholder
        self.lookup: dict[str, str] = None

    def _hash(self, smiles: list[str]) -> dict[str, str]:
        """Hashing for amateurs.
        Goal: Get a short, valid, and hopefully unique filename for each molecule."""
        self.lookup = {smile: str(uuid.uuid4())[:8] for smile in smiles}
        return self

    def get_path(self) -> str:
        return self.path

    def get_molecule_filesnames(self):
        return self.lookup

    def plot(self, smiles: Union[list[str], str]):
        """Plot smiles as 2d molecules and save to `self.path/subfolder/*.svg`."""
        self._hash(smiles)

        for k, v in self.lookup.items():
            fname = self.outfolder / f"{v}.svg"
            mol = Chem.MolFromSmiles(k)
            # Plot
            drawer = Draw.rdMolDraw2D.MolDraw2DSVG(600, 300)
            opts = drawer.drawOptions()
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            p = drawer.GetDrawingText()

            with open(fname, "w") as f:
                f.write(p)

        return self
    

class RxnDrawer:
    """Draws reactions as images."""

    def __init__(self, path: Optional[str], subfolder: str = "assets"):

        # Init outfolder
        if not (path is not None and Path(path).exists()):
            raise NotADirectoryError(path)
        self.outfolder = Path(path) / subfolder
        self.outfolder.mkdir(exist_ok=1)

        # Placeholder
        self.lookup: dict[str, str] = None

    def _hash(self, smirks: list[str]) -> dict[str, str]:
        """Hashing for amateurs.
        Goal: Get a short, valid, and hopefully unique filename for each reaction."""
        self.lookup = {smirk: str(uuid.uuid4())[:8] for smirk in smirks}
        return self

    def get_path(self) -> str:
        return self.path

    def get_reaction_filesnames(self):
        return self.lookup

    def plot(self, smirks: Union[list[str], str]):
        """Plot smirks and save to `self.path/subfolder/*.png`."""
        self._hash(smirks)

        for k, v in self.lookup.items():
            fname = self.outfolder / f"{v}.png"
            # Plot      
            rxn = AllChem.ReactionFromSmarts(k)
            d2d = Draw.MolDraw2DCairo(1600, 600)
            d2d.DrawReaction(rxn, highlightByReactant=True)
            p = d2d.GetDrawingText()                

            with open(fname, "wb+") as f:
                f.write(p)

        return self    


if __name__ == "__main__":
    pass
