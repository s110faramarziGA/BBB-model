"""A class for decoding chemical information from the Simplified Molecular Input Line Entry System (SMILES)"""

import csv
from collections import (
    Counter, 
    defaultdict
)
import itertools
import os
from typing import Literal

from .vertex import Vertex
from .edge import Edge
from .constants import (
    AMINO_ACID_REGEX,
    ATOM_REGEX,
    BOND_REGEX,
    CHARGE_REGEX,
    DIGIT_REGEX,
    ELECTRON_BOND_COUNTS,
    PARENTH_REGEX,
    REQUIRED_VALENCE_COUNTS,
    SMILES_REGEX,
)


class Molecule():
    """ A python class for the SMILES.

        Derives a Simple Connected Undirected Molecular Graph.
        Determines the number of aromatic and non aromatic rings for organic molecules.
        Determines the frequency of each unique ring-classified functional group.
        
        Parameters
        ----------
        smiles : str
            A hydrogen-suppressed SMILES code 
        name : str
            A name identifier for the molecule or functional group
        type : Literal["mol", "fg"]
            An organic molecule (mol) or functional group (fg) SMILES code

        Returns
        -------
        Molecule
            An instance of a molecule object

        Example
        -------
            >>> mol = Molecule("O=C1NCCCN1", name="APYFEB01", type="mol")
            >>> fg = Molecule("[R]C(=O)O[R]", name="Ester" , type="fg")

    """

    def __init__(self, 
        smiles: str, 
        name: str = "", 
        type: Literal["mol", "fg"] = "mol"
    ):
        """ Generates a software molecule graph of any SMILES defined molecule. 
            Generates ring and functional group data for organic molecules
        """

        ##### Input Data #####
        self.smiles: list[str] = SMILES_REGEX.findall(smiles)
        """The list of all smiles code symbols, with charges attached to atoms as needed, according to the SMILES_REGEX capture groups"""

        self.atoms: list[str] = ATOM_REGEX.findall(smiles)
        """The list of all smiles code atoms, inclusive of charges, according to the ATOM_REGEX capture groups"""

        self.name: str = name
        """The name identifier for the smiles code"""

        assert ['[', ']'] not in self.atoms

        ##### Software Molecule Graph (Graph Theory) #####
        self.vertices: "list[Vertex]" = self.createVertices()
        """The list of vertices of the molecular graph"""

        self.order: int = len(self.vertices)
        """The number of vertices of the molecular graph"""

        self.edges: "list[Edge]" = self.createEdges()
        """The list of edges of the molecular graph"""

        self.size: int = len(self.edges)
        """The number of edges of the molecular graph"""

        assert self.order == len(self.atoms)

        ##### Ring Data #####
        self.ring_atoms: set[int]
        """The vertex indices which are apart of a ring structure"""

        self.aromatic_ring_count: int
        """The number of aromatic rings in the molecule"""

        self.non_aromatic_ring_count: int
        """The number of non aromatic rings in the molecule"""

        self.ring_atoms, self.aromatic_ring_count, self.non_aromatic_ring_count = self.createRings()

        self.total_ring_count: int = self.aromatic_ring_count + self.non_aromatic_ring_count
        """The total number of rings in the molecule"""

        self.total_ring_atom_count: int = len(self.ring_atoms)
        """The total number of atoms apart of rings in the molecule"""

        self.total_aromatic_atoms: int = len([symbol for symbol in self.atoms if symbol.islower()])
        """The total number of aromatic atoms in the molecule"""

        self.total_non_aromatic_atoms: int = self.total_ring_atom_count-self.total_aromatic_atoms
        """The total number of non aromatic atoms in the molecule"""
    
        ##### Atom Counts #####
        self.atom_freq: dict[str, int] = Counter([v.symbol for v in self.vertices])
        """The frequency of each atom in the molecule"""
    
        ##### Miscellaneous Molecular Data #####
        self.amino_acid: bool = len(AMINO_ACID_REGEX.findall(smiles)) != 0 
        """The assertion of a present amino acid in the molecule"""

        ##### Functional Groups #####
        self.functional_groups_all: dict[str, int]
        """The frequency of each functional group inclusive of overlapped functional groups"""

        self.functional_groups_exact: dict[str, int]
        """The frequency of each functional group exclusive of overlapped functional groups"""
        
        self.functional_groups_all, self.functional_groups_exact = self.createFunctionalGroups() if type == "mol" else ({}, {})


    def createVertices(self) -> "list[Vertex]":
        """Creates the vertices of a software molecule graph using a hydrogen-suppressed SMILES code.

            Creates a vertex object for each unique atomic symbol, inclusive of their charge, and 
            computes their valence electrons required to set up :ref:`hidden-hydrogens-computation-ref`.
            
            Returns
            -------
            list[Vertex]
                A list of Vertex objects

            Notes
            -----
            This algorithm iterates over each unique atomic symbol in the SMILES code, inclusive of their charge, and creates 
            a vertex object for each one. The constructor parameters of integer index, atomic symbol with charge, aromatic conditional 
            (if the atomic symbol is lower case), and number of valence electrons required to be provided by edge bonds to fulfill the atom's preferred 
            electronic configuration are all input into each individual ``vertex`` constructor. Each ``vertex`` is appended to the list of ``vertices``
            until all unique atomic symbols in the SMILES code have been processed. The :py:attr:`vertex.Vertex.valence_electrons_required` computation 
            in this method sets up :ref:`Hidden Hydrogen Computation <hidden-hydrogens-computation-ref>` for each ``vertex`` in ``vertices`` to take place 
            during the :py:meth:`molecule.Molecule.createEdges` method. Vertex indices are assigned in ascending order for unique atomic symbols left to right.

            | `Algorithm Variables Reference`
            | ``vertices``  (list[Vertex]):     cumulative list of vertex objects
            | ``vertex``    (Vertex):           vertex object per unique atomic symbol
            | ``charge``    (str):              charge symbol attached to an atom symbol

        """

        ##### Vertex List and Objects #####
        vertices: list[Vertex] = []
        vertex: Vertex = Vertex()
        charge: str = ""

        ##### Atom Symbols Loop #####
        for index, symbol in enumerate(self.atoms):

            ##### Charge Symbol Case #####
            if CHARGE_REGEX.search(symbol):
                charge = CHARGE_REGEX.search(symbol).group()               # type: ignore
                symbol = ''.join([char for char in symbol if charge != char])

            ##### Vertex Object Construction #####
            vertex = Vertex(
                index=index, 
                symbol=symbol.upper() + charge, 
                is_aromatic=symbol.islower(), 
                valence_electrons_required=REQUIRED_VALENCE_COUNTS[symbol.upper()],
                charge=charge
            )

            ##### Reset Charge #####
            charge = ""

            ##### Append Vertex Object #####
            vertices.append(vertex)

        ##### Return vertices #####
        return vertices
            
    def createEdges(self) -> "list[Edge]":
        """Creates the edges and the vertex degrees of a software molecule graph using a hydrogen-suppressed SMILES code.
        
            Creates an edge object for every bond between two atomic symbols 
            and computes the hidden hydrogen count for each vertex. 

            Returns
            -------
            list[Edge]
                A list of Edge objects

            Notes
            -----
                View :ref:`edges-algorithm-ref` under :ref:`implementation-ref` for algorithm details.
                    
                | `Algorithm Variables Reference`
                | ``atom_index``                (int):                          an index counter for each unique atomic symbol (synonomous with vertex index counter)
                | ``match_index``               (int):                          a unique atomic symbol index variable for :ref:`direct-edge-ref` and :ref:`indirect-parenthetical-edge-ref` pairing
                | ``edge_index``                (int):                          an index counter variable for each unique edge
                | ``open_ring_table``           (dict[str, int]):               a dictionary for :ref:`open-rings-ref` that key-value pairs start digit value with start atom index for :ref:`Indirect Number Edges <indirect-number-edge-ref>`
                | ``parenth_start_atom_stack``  (list[int]):                    an atom index stack pushed with parenthetical start atom index upon open parenthesis and popped into ``match_index`` upon close parenthesis
                | ``bond``                      (Literal["", "=", "#"]):        the most recently viewed bond symbol in the iteration, cleared after use in a :ref:`direct-edge-ref`
                | ``edges``                     (list[Edge]):                   cumulative list of edge objects

        """

        ##### Algorithm Variables #####
        atom_index: int = 0
        match_index: int = 0
        edge_index: int = 0 
        open_ring_table: "dict[str, int]" = {}
        parenth_start_atom_stack: list[int] = []
        bond: Literal["", "=", "#"] = ""
        edges: "list[Edge]" = []
        
        ##### Algorithm Implementation #####
        for i,symbol in enumerate(self.smiles[1:]):

            ##### Atom Symbol Case #####
            if ATOM_REGEX.match(symbol):
                atom_index+=1
                edge_atoms = [self.vertices[match_index], self.vertices[atom_index]]
                new_edge = Edge(edge_atoms, bond, edge_index)
                edge_index+=1
                edges.append(new_edge)
                match_index = atom_index
                bond = ""

            ##### Bond Symbol Case #####
            if BOND_REGEX.match(symbol):
                bond = symbol               # type: ignore

            ##### Digit Symbol Case #####
            if DIGIT_REGEX.match(symbol):
                if symbol in open_ring_table:
                    ring_atom_index = open_ring_table.pop(symbol)
                    edge_atoms = [self.vertices[ring_atom_index], self.vertices[atom_index]]
                    new_edge = Edge(edge_atoms, "", edge_index)
                    edge_index+=1
                    edges.append(new_edge)
                else:
                    open_ring_table[symbol] = atom_index 

            ##### Parenthesis Symbol Case #####
            if PARENTH_REGEX.match(symbol):
                if symbol == '(':    
                    # double parenthetical groups [i.e. C(C)(C)] will re-append the match index 
                    if self.smiles[1:][i-1] == ')':
                        parenth_start_atom_stack.append(match_index)
                    else:
                        parenth_start_atom_stack.append(atom_index)
                else:
                    match_index = parenth_start_atom_stack.pop()

        
        ##### Algorithm Check #####
        assert not parenth_start_atom_stack
        assert not open_ring_table
        assert not bond


        ##### Set Vertex Degrees #####
        for vertex in self.vertices:

            ##### R Vertex Degree #####
            if vertex.symbol == 'R':
                vertex.implicit_degree = 0
                vertex.explicit_degree = 1
                vertex.total_degree = 1
                continue

            ##### Core Vertex Degree #####
            total_edges = [edge for edge in edges if vertex.index in edge.indices]
            explicit_valence_electrons = sum([ELECTRON_BOND_COUNTS[edge.bond_type] for edge in total_edges])        
            implicit_valence_electrons = vertex.valence_electrons_required - explicit_valence_electrons            # number of hydrogens
            vertex.explicit_degree = len(total_edges)
            vertex.implicit_degree = implicit_valence_electrons
            vertex.total_degree = vertex.implicit_degree + vertex.explicit_degree

        ##### Algorithm Results #####
        return edges

        
    def createRings(self):
        """ Creates the rings of the software molecule graph from a SMILES code by identifying them.
        
            Determines the number of aromatic and non-aromatic rings, and 
            distinguishes all atoms as aromatic, non-aromatic, or non-cyclic.

            Returns
            -------
            None

            Notes
            -----
            View the :ref:`rings-algorithm-ref` under :ref:`implementation-ref` for algorithm details.
            
            | `Algorithm Variables Reference`
            | ``ring_index``              (int):                  an index counter for unique :ref:`Like Number Pairs <like-number-pair-ref>`
            | ``p_group_counter``         (int):                  an index counter for :ref:`parenthetical-groups-ref`
            | ``parenth_group_stack``     (list[int]):            a stack of ``p_group_counter`` for *open* parenthetical groups (always has root group 0)
            | ``open_ring_table``         (dict[str, int]):       a dictionary of key ring digit value to value ``ring_index`` for :ref:`open-rings-ref`
            | ``ring_info``               (dict[int, list[int]]): a dictionary of key ``ring_index`` to value set of allowable parenthetical groups for :ref:`ring-assigned-parenthetical-groups-ref` 
            | ``atom_index``              (int):                  an index counter for each unique atomic symbol (synonomous with vertex index counter)
            | ``ring_stack``              (list[int]):            a stack of ``ring_index`` for the order of :ref:`open-rings-ref` (``ring_stack[-1]`` is the most recently opened ring)
            | ``ring_set``                (dict[int, list[int]]): a dictionary of key ``ring_index`` to value set of atomic indices
            | ``ring_p_groups``           (set[int]):             a set of :ref:`Open Ring <open-rings-ref>` parenthetical group indices
            | ``ring_atom_indices``       (set[int]):             the set of atom indices apart of rings (aromatic or non-aromatic)
            | ``aromatic_ring_count``     (int):                  the number of aromatic rings
            | ``non_aromatic_ring_count`` (int):                  the number of non-aromatic rings 
        """

        ########## Parenthetical Groups Preparation ##########

        ##### Preparation Variables #####
        ring_index: int = 0
        p_group_counter: int = 0
        parenth_group_stack: list[int] = [0]
        open_ring_table: dict[str, int] = {}
        ring_info: dict[int, list[int]] = {}

        ##### Preparation Implementation #####
        for symbol in self.smiles[1:]:

            ##### Digit Symbol Case #####
            if DIGIT_REGEX.match(symbol):

                if symbol in open_ring_table:
                    open_ring_table.pop(symbol)

                else:
                    open_ring_table[symbol] = ring_index
                    ring_info[ring_index] = [parenth_group_stack[-1]]
                    ring_index+=1

            ##### Parenthesis Symbol Case #####
            if PARENTH_REGEX.match(symbol):

                if symbol == '(':
                    p_group_counter+=1
                    parenth_group_stack.append(p_group_counter)
                    for ring_idx in open_ring_table.values():
                        ring_info[ring_idx].append(p_group_counter)

                else:
                    closing_p_group = parenth_group_stack.pop(-1)
                    for ring_idx in open_ring_table.values():
                        ring_info[ring_idx] = [p_group for p_group in ring_info[ring_idx] if p_group != closing_p_group]

        ##### Preparation Check #####
        assert not open_ring_table
        assert parenth_group_stack == [0]

        ##### Preparation Results #####
        # print(ring_info)

        ########## Algorithm Implementation ##########

        ##### Algorithm Variables #####
        ring_index = 0
        p_group_counter = 0
        atom_index: int = 0
        ring_stack: list[int] = []
        ring_set: dict[int, list[int]] = {}
        ring_p_groups: set[int] = set()
        ring_atom_indices: set[int] = set()

        ##### Algorithm Implementation #####
        for symbol in self.smiles[1:]:

            ##### Atom Symbol Case #####
            if ATOM_REGEX.match(symbol):
                atom_index+=1

                if open_ring_table:

                    if parenth_group_stack[-1] in ring_p_groups:
                        ring_atom_indices.add(atom_index)

                    if parenth_group_stack[-1] in ring_info[ring_stack[-1]]:
                        ring_set[ring_stack[-1]].append(atom_index)

            ##### Digit Symbol Case #####
            if DIGIT_REGEX.match(symbol):

                if symbol in open_ring_table:

                    close_ring_index = open_ring_table.pop(symbol)

                    if open_ring_table:
                        prev_ring_index = ring_stack[ring_stack.index(close_ring_index)-1]
                        p_end_group = ring_info[close_ring_index][-1]
                        if p_end_group in ring_info[prev_ring_index]:
                            ring_set[prev_ring_index].append(atom_index)

                    if not atom_index in ring_set[close_ring_index]:
                        ring_set[close_ring_index].append(atom_index)
                else:
                    open_ring_table[symbol] = ring_index
                    ring_set[ring_index] = [atom_index]
                    ring_atom_indices.add(atom_index)
                    ring_index+=1
                
                ring_stack = list(open_ring_table.values())
                ring_p_groups = set(
                    itertools.chain.from_iterable(
                        [p_groups for ring_idx, p_groups in ring_info.items() if ring_idx in open_ring_table.values()]
                    )
                )


            ##### Parenthesis Symbol Case #####
            if PARENTH_REGEX.match(symbol):

                if symbol == '(':
                    p_group_counter+=1
                    parenth_group_stack.append(p_group_counter)

                else:
                    parenth_group_stack.pop(-1)


        ##### Algorithm Check #####
        assert not open_ring_table
        assert parenth_group_stack == [0]

        ##### Algorithm Results #####
        # print(ring_set)
        # print(ring_atom_indices)

        ########## Algorithm Collection ##########

        ##### Collection Variables #####
        aromatic_ring_count: int = 0
        non_aromatic_ring_count: int = 0

        ##### Collection 1: Ring Counts #####
        for (ring_idx, atom_indices) in ring_set.items():

            if len([v for v in atom_indices if self.vertices[v].ring_type == "aromatic"]) == len(atom_indices):
                aromatic_ring_count+=1
            else:
                non_aromatic_ring_count+=1

        ##### Collection 2: Atom Ring Types #####
        for atom_index in ring_atom_indices:
            if self.vertices[atom_index].ring_type == "non-cyclic":
                self.vertices[atom_index].ring_type = "non-aromatic"

        ##### Collection Check #####
        assert len(ring_info.keys()) == (aromatic_ring_count + non_aromatic_ring_count)

        ##### Collection Results #####
        return (
            ring_atom_indices,  
            aromatic_ring_count,
            non_aromatic_ring_count,
        )


    def createFunctionalGroups(self):
        """Determine the frequency of the unique ring classified functional groups given a list of identifiable functional groups using the software molecule graph format.
        
            Loops over a set of identifiable functional groups, generates each graph, and executes the DFS algorithm using all possible starting vertex pairs 
            in the molecule and functional group. Adds ring-classification to each functional group, applies accuracy filters and returns the 
            frequency counts in the overlap inclusive and exclusive formats.

            Returns
            -------
            all_fgs_dict
                A frequency dictionary with the count of each functional group inclusive of overlapped functional groups

            exact_fgs_dict
                A frequency dictionary with the count of each functional group exclusive of overlapped functional groups

            Notes
            -----
                View :ref:`functional-groups-algorithm-ref` under :ref:`implementation-ref` for algorithm details
                Only "mol" type given in constructor calls this function

            | `Algorithm Variables Reference`
            | ``all_fgs``                 (list[Molecule]):             a list of all functional group matches under the Molecule class type, hierarchically filtered
            | ``fg``                      (Molecule):                   a functional group graph template built from its hydrogen-suppressed SMILES code
            | ``fg_matches``              (list[dict[int,int]]):        a list of ``matched_indices`` results from the DFS algorithm 
            | ``like_vertex_pairs``       (dict[int, list[Vertex]]):    a dicitonary of *core* functional group vertex index to all :ref:`Like Vertex Paired <like-vertex-pair-ref>` organic molecule vertex indices
            | ``fg_vertex``               (Vertex):                     a *core* functional group vertex which will begin the DFS algorithm
            | ``fg_match``                (Molecule):                   a Molecule generated functional group match with overwritten :ref:`Like Vertex Paired <like-vertex-pair-ref>` organic moleulce vertex indices
            | ``exact_fgs``               (list[int]):                  a list of matches after ``all_fgs`` is overlap filtered
        """

        ##### All Functional Group Matches #####
        all_fgs: list[Molecule] = []

        ##### Functional Groups Smiles Codes CSV File #####
        functional_group_smiles_codes_csv_file = open(os.path.dirname(__file__) + "/data/functional_group_smiles_codes.csv")

        ##### Functional Group Loop #####
        for (fg_smiles, fg_name) in csv.reader(functional_group_smiles_codes_csv_file, delimiter=",", skipinitialspace=True):

            ##### Functional Group Graph Template #####
            fg: Molecule = Molecule(fg_smiles, fg_name, "fg")

            ##### Functional Group Matches #####
            fg_matches: list[dict[int,int]] = []

            ##### Functional Group Mol Vertex Start Locations #####
            like_vertex_pairs: dict[int, list[Vertex]] = {
                fg_vertex.index: [
                    mol_vertex for mol_vertex in self.vertices 
                    if mol_vertex.symbol == fg_vertex.symbol and 
                    mol_vertex.total_degree == fg_vertex.total_degree
                ]
                for fg_vertex in fg.vertices if fg_vertex.symbol != 'R'
            }

            ##### Functional Group Mol Vertex Start Locations Loop #####
            for fg_vertex_index, matched_mol_vertices in like_vertex_pairs.items():

                ##### Functional Group Start Vertex #####
                fg_vertex: Vertex = fg.vertices[fg_vertex_index]

                ##### Molecule Start Vertex Locations Loop #####
                for mol_vertex in matched_mol_vertices:

                    ##### Functional Group DFS Match Algorithm #####
                    fg_matched_atoms, _, _ = self.DFS(fg, fg_vertex, mol_vertex, [], [])

                    ##### Functional Group Match Case #####
                    if (
                        len(fg_matched_atoms) == len([vertex for vertex in fg.vertices if vertex.symbol != 'R'])
                        and
                        not set(fg_matched_atoms.values()) in [set(match.values()) for match in fg_matches]
                    ):
                        ##### Unique Matched Functional Group Add ##### 
                        fg_matches.append(fg_matched_atoms)


            ##### Functional Group Molecular Object Creations #####
            for fg_matched_atoms in fg_matches:
                
                ##### Functional Group Extraction #####
                fg_match: Molecule = Molecule(fg_smiles, fg_name, "fg")
                for (fg_atom_index, om_atom_index) in fg_matched_atoms.items():
                    fg_match.vertices[fg_atom_index].index = self.vertices[om_atom_index].index
                    fg_match.vertices[fg_atom_index].ring_type = self.vertices[om_atom_index].ring_type

                ##### Ring Classification #####
                aromatic_tally: int = len([fg_vertex for fg_vertex in fg_match.vertices if fg_vertex.symbol != 'R' and fg_vertex.ring_type == "aromatic"])
                non_aromatic_tally: int = len([fg_vertex for fg_vertex in fg_match.vertices if fg_vertex.symbol != 'R' and fg_vertex.ring_type == "non-aromatic"])
                if aromatic_tally != 0 or non_aromatic_tally != 0:
                    nomenclature: str = "Aromatic " if aromatic_tally >= non_aromatic_tally else "Non Aromatic "
                    fg_match.name = nomenclature + fg_match.name
                
                ##### Match Add #####                        
                all_fgs.append(fg_match)


        ##### Hierarchical Functional Group Filter #####
        all_fgs: list[Molecule] = self.hierarchyFilter(all_fgs)

        ##### Overlapping Functional Group Filter #####
        exact_fgs: list[Molecule] = self.overlapFilter(all_fgs)

        ##### All Functional Group Counts #####
        all_fgs_dict = defaultdict(int)
        for fg in all_fgs:
            all_fgs_dict[fg.name] += 1

        ##### Exact Functional Group Counts #####
        exact_fgs_dict = defaultdict(int)
        for fg in exact_fgs:
            exact_fgs_dict[fg.name] += 1

        ##### Algorithm Results #####
        return (all_fgs_dict, exact_fgs_dict)

    def DFS(self, 
        fg: "Molecule", 
        fg_vertex: Vertex, 
        mol_vertex: Vertex, 
        used_mol_edges: "list[int]", 
        used_fg_edges: "list[int]"
    ) -> 'tuple[dict[int, int], list[int], list[int]]':
        """Searches an organic molecule software graph for the presence of a functional group sub-graph using a recursive depth first search and backtracking algorithm.
        
            Parameters
            ----------
            fg : Molecule
                The functional group graph being searched for

            fg_vertex : Vertex
                The current functional group vertex

            mol_vertex : Vertex
                The current molecular vertex 

            used_mol_edges : list[int]
                The list of molecular edge indices that have already been paired with functional group edges

            used_fg_edges
                The list of functional group edge indices that have already been paired with molecular edges

            Returns
            -------
            matched_path_atoms: dict[int, int]
                A recusivly cumulative dictionary of matched vertex pairs by index in the form functional_group_index : molecular_index
            
            matched_mol_path_edges: list[int]
                A recursivly cumulative list of used molecular edges during a search path

            matched_fg_path_edges: list[int]
                A recursivly cumulative list of used functional group edges during a search path

            Notes
            -----
                View :ref:`depth-first-search-ref` under :ref:`implementation-ref` for algorithm details.

            | `Algorithm Variables Reference`
            | ``matched_indices``         (dict[int,int]):        a backtracking-cumulative dictionary of key functional group vertex index to value organic molecule vertex index for :ref:`like-vertex-pair-ref`
            | ``fg_core_edges``           (list[Edge]):           a list of functional group edges to unfulfilled *core* functional group vertices
            | ``om_edges``                (list[Edge]):           a list of organic molecule edges to un-used organic molecule vertices
            | ``fg_complement_vertex``    (Vertex):               an *unfulfilled* core functional group vertex 
            | ``path``                    :                       a recursive call to the DFS algorithm which searches into new vertices and accumulates backtracking data
            | ``matched_path_atoms``      (dict[int,int]):        a backtracking-cumulative version of ``matched_indices``
            | ``matched_mol_path_edges``  (list[int]):            a backtracking-cumulative version of ``used_mol_edges``
            | ``matched_fg_path_edges``   (list[int]):            a backtracking-cumulative version of ``used_fg_edges``
        """

        ##### New Atom-Pair Backtrack Variable #####
        matched_indices = {fg_vertex.index: mol_vertex.index}

        ##### Edge Sets #####
        fg_core_edges = [edge for edge in fg.edges if fg_vertex.index in edge.indices and not edge.index in used_fg_edges and not 'R' in edge.symbols]
        om_edges = [edge for edge in self.edges if mol_vertex.index in edge.indices and not edge.index in used_mol_edges]

        ##### Implicit Degree Validation #####
        if fg_vertex.implicit_degree != 0 and mol_vertex.implicit_degree < fg_vertex.implicit_degree:
            return ({}, [], [])

        ##### Functional Group End Graph Boundary Case #####
        if not fg_core_edges:
            return ({fg_vertex.index: mol_vertex.index}, used_mol_edges, used_fg_edges)

        ##### Functional Group Core Edge Set Searching #####
        for fg_edge in fg_core_edges:

            ##### Complementary Functional Group Vertex #####
            fg_complement_vertex = fg_edge.complement_vertex(fg_vertex.index)

            ##### Organic Molecule Edge Set Match Attempts #####
            for om_edge in om_edges:

                ##### Unused Organic Molecule Edges #####
                if om_edge.index not in used_mol_edges:

                    ##### Complementary Molecule Vertex #####
                    om_corresponding_vertex = om_edge.complement_vertex(mol_vertex.index)

                    ##### Edge Structure & Complementary Vertex Degree Validation #####
                    if (
                        om_edge == fg_edge 
                        and 
                        om_corresponding_vertex.total_degree == fg_complement_vertex.total_degree
                    ):

                        ##### DFS Recursion #####
                        path = self.DFS(fg, fg_complement_vertex, om_corresponding_vertex, used_mol_edges + [om_edge.index], used_fg_edges + [fg_edge.index])

                        ##### Backtrack Collection #####
                        if all(path):

                            ##### Backtrack Unpacking #####
                            matched_path_atoms, matched_mol_path_edges, matched_fg_path_edges = path

                            ##### Atom Unpacking #####
                            for matched_fg_atom, matched_mol_atom in matched_path_atoms.items():
                                matched_indices[matched_fg_atom] = matched_mol_atom
                            
                            ##### Molecule Edge Unpacking #####
                            for om_edge_index in matched_mol_path_edges:
                                used_mol_edges.append(om_edge_index)
                            
                            ##### Functional Group Edge Unpacking #####
                            for fg_edge_index in matched_fg_path_edges:
                                used_fg_edges.append(fg_edge_index)
                            
                            ##### Satisfied Functional Group Edge #####
                            break

            ##### Unsatisfied Functional Group Edge #####
            else:
                return ({}, [], [])

        ##### All Functional Group Core Edges Satisfied #####
        return (matched_indices, used_mol_edges, used_fg_edges)

    def hierarchyFilter(self, all_fgs: "list[Molecule]") -> "list[Molecule]":
        """Identifies and filters hierarchically related functional group matches.

            Uses the theory of :ref:`hierarchical-functional-groups-ref` to identify 
            the hierarchical functional groups from ``all_fgs`` matches, then identifies
            the most accurate group out of the hierarchy using the hidden hydrogen exactness test 
            for exact hydrogen-sensetive R vertices exhibited in the organic molecule. 

            Parameters
            ----------
            all_fgs : list[Molecule]
                A list of matched molecular functional group objects

            Returns
            -------
            list[Molecule]
                A list of matched molecular functional groups filtered hierarchically for the most accurate group

            Notes
            -----
                View :ref:`hierarchy-filter-implementation-ref` under :ref:`implementation-ref` for algorithm details

            | `Algorithm Variables Reference`
            | ``eval_indices``  (list[int]):   a list of index positions in ``all_fgs`` where hierarchical functional groups are
            | ``skip_indices``  (list[int]):   a list of index positions to remove from ``all_fgs``
        
        """

        ##### Matches List Evaluation Indices #####
        eval_indices: set[int] = set()

        ##### Hierarchical Functional Group Identification #####
        for i, fg in enumerate(all_fgs):
            for j, fg_compare in enumerate(all_fgs):
                if i == j:
                    continue
                if (
                    set([edge for edge in fg.edges if not 'R' in edge.symbols]) == set([edge for edge in fg_compare.edges if not 'R' in edge.symbols]) 
                    and 
                    set([fg_vertex.index for fg_vertex in fg.vertices if 'R' not in fg_vertex.symbol]) == set([fg_vertex.index for fg_vertex in fg_compare.vertices if 'R' not in fg_vertex.symbol])
                ):
                    eval_indices.add(i)
                    eval_indices.add(j)
        
        ##### Indices To-Be Skipped From Matches List #####
        skip_indices: set[int] = set()

        ##### Hierarchical Accuracy Selection #####
        for i in eval_indices:
            fg: Molecule = all_fgs[i]
            for core_atom in [fg_vertex for fg_vertex in fg.vertices if fg_vertex.symbol != 'R']:
                if (
                    core_atom.explicit_degree == self.vertices[core_atom.index].explicit_degree 
                    and 
                    core_atom.implicit_degree == self.vertices[core_atom.index].implicit_degree
                ):
                    continue
                else:
                    skip_indices.add(i)
                    break

        ##### Apply Skips For Accurate Results #####
        return [fg for i, fg in enumerate(all_fgs) if not i in skip_indices]
        
    def overlapFilter(self, all_fgs: "list[Molecule]") -> "list[Molecule]":
        """Identifies and filters overlapping functional group matches.
        
            Uses the theory of :ref:`overlapping-functional-groups-ref` to identify and remove 
            smaller functional groups overlapped with larger functional groups from the 
            list of matches ``all_fgs``

            Parameters
            ----------
            all_fgs : list[Molecule]
                A list of matched molecular functional group objects

            Returns
            -------
            list[Molecule]
                A list of matched molecular functional groups with overlapped functional group occurences removed


        """

        ##### Indices To-Be Skipped From Matches List #####
        skip_indices: set[int] = set()

        ##### Overlapping Functional Group Identification and Accuracy Selection #####
        for i, fg in enumerate(all_fgs):
            for fg_compare in all_fgs:
                if (
                    len([fg_vertex for fg_vertex in fg.vertices if fg_vertex.symbol != 'R']) < 
                    len([fg_vertex for fg_vertex in fg_compare.vertices if fg_vertex.symbol != 'R'])
                ):
                    if set([fg_vertex.index for fg_vertex in fg.vertices if fg_vertex.symbol != 'R']).issubset(set([fg_vertex.index for fg_vertex in fg_compare.vertices if fg_vertex.symbol != 'R'])):
                        skip_indices.add(i)

        ##### Apply Skips For Accurate Results #####
        return [fg for i, fg in enumerate(all_fgs) if not i in skip_indices]

    def __str__(self):
        """String Representation of a Molecule"""
        return ''.join(self.smiles)

    def __repr__(self):
        """General Representation of a Molecule (Same as String Representation)"""
        return ''.join(self.smiles)