from glob import glob
import numpy as np
from tqdm import tqdm
from hashlib import sha1

# Two possible quadratic forms for T=1 unimodular lattice
OMEGA_ODD  = np.array([[1, 0], [0,-1]])
OMEGA_EVEN = np.array([[0, 1], [1, 0]])


class Helper():

    def __init__(self, T, folder_data, folder_branching_rules = None):

        if T not in [0, 1]:
            raise ValueError('T must be either zero or one.')

        self.T = T
        self.folder_data = folder_data
        self.folder_branching_rules = folder_branching_rules

        self.load_vertices()
        self.load_models()
        if folder_branching_rules is not None:
            self.load_branching_rules()

    def load_vertices(self):
        """Loads all vertex data."""

        # Read in the upper bounds on Δ
        print(f'Vertices\n  Reading in upper bounds on Δ from {self.folder_data}/groups-Δmax.txt')
        self.Δ_upper_bounds = {'SU': [], 'SO': [], 'Sp': [], 'EFG': []}
        with open(self.folder_data + '/groups-Δmax.txt', 'r') as file:
            for line in file:
                group_type, Δmax_data = line[:-1].split('\t')
                Δmax_data = [xx.split(': ') for xx in Δmax_data[1:-1].split(', ')]
                Δmax_data = [[int(n), int(Δmax)] for n, Δmax in Δmax_data]
                
                for n, Δmax in Δmax_data:
                    if group_type == 'A':
                        self.Δ_upper_bounds['SU'].append([n+1, Δmax])
                    elif group_type == 'B':
                        self.Δ_upper_bounds['SO'].append([2*n+1, Δmax])
                    elif group_type == 'C':
                        self.Δ_upper_bounds['Sp'].append([n, Δmax])
                    elif group_type == 'D':
                        self.Δ_upper_bounds['SO'].append([2*n, Δmax])
                    else:
                        self.Δ_upper_bounds['EFG'].append([n, Δmax])

        files = glob(self.folder_data + '/vertices/*-vertices.tsv')
        print(f'  Loading vertices from {len(files)} files in the folder {self.folder_data}/vertices')

        # self.vertices is a dictionary with group IDs (e.g. A03, E08) for keys
        # and each item is itself a dictionary of vertices with IDs for keys
        self.vertices = dict()
        for file in tqdm(files, ascii=True, desc='  Files'):
            # Get group from filename /X##-vertices.tsv
            group_ID = file[-16:-13]
            with open(file, 'r') as file_in:
                group_vertices = [Vertex(line) for line in file_in]
                self.vertices[group_ID] = {vertex.ID: vertex for vertex in group_vertices}

        # Identify SO(8) vertices which are triality-invariant and save their IDs
        self.triality_invariant_vertex_IDs = set()
        if 'D04' in self.vertices:
            for vertex_ID, vertex in self.vertices['D04'].items():
                if vertex.is_triality_invariant():
                    self.triality_invariant_vertex_IDs.add(vertex_ID)

        total_vertices = sum([len(aa) for aa in self.vertices.values()])
        print(f'DONE. Loaded a total of {total_vertices:,} vertices for {len(self.vertices)} simple groups.\n')

    def load_models(self):
        """Loads all model data."""

        files = glob(self.folder_data + '/anomaly-free-models/models-SNT=(*).tsv')
        print(f'Models\n  Loading models from {len(files)} files in the folder {self.folder_data}/anomaly-free-models')

        # self.models is a list of models (including "empty" model with V=0)
        self.models = [Model(self.T)]
        for file in tqdm(files, ascii=True, desc='  Files'):
            with open(file, 'r') as file_in:
                models_new = [Model(self.T, line) for line in file_in]
                self.models.extend(models_new)

        # Sort by dim/rank of gauge group
        sort_data = np.array([[model.gauge_group_rank, model.gauge_group_dim] for model in self.models])
        order = np.lexsort(sort_data.T)
        self.models = [self.models[ii] for ii in order]

        print('  Tallying vertex usage...')
        for group_vertices in self.vertices.values():
            for vertex in group_vertices.values():
                vertex.times_used = 0
        for model in tqdm(self.models, ascii=True, desc='    Model'):
            for vertex_ID in model.vertex_IDs:
                group_ID = vertex_ID[4:7]
                self.vertices[group_ID][vertex_ID].times_used += 1

        # Also save models in a dictionary with hashes for keys. This is useful to have
        # when trying to find the corresponding models produced during Higgsing
        self.models_by_hash = {hash(model): model for model in self.models}
        print('  Pre-computing...')
        self.models_by_hash = dict()
        for model in tqdm(self.models, ascii=True, desc='    Hashes'):
            self.models_by_hash[model.get_hyper_hash()] = model
        for model in tqdm(self.models, ascii=True, desc='    Triality'):
            model.triality_equivalent = self.find_triality_equivalent(model)
        print(f'DONE. Loaded a total of {len(self.models):,} models.\n')

    def load_branching_rules(self):
        """Loads all branching rules."""

        files = glob(self.folder_branching_rules + '/*/*.tsv')
        print(f'Branching rules\n  Loading branching rules from {len(files)} files in the folder {self.folder_branching_rules}')

        self.branching_rules = dict()
        for file in tqdm(files, ascii=True, desc='  Files'):
            # Extract groups from filename
            group_from, groups_to = file[:-4].split('\\')[-1].split('-')[:2]
            groups_to = tuple(groups_to.split('x'))

            # Only if groups are present
            if group_from not in self.vertices \
                or any([gr not in self.vertices and gr not in ['A01', 'A02', 'C02'] for gr in groups_to]):
                continue

            # Read in irrep branching rules
            rule_data = dict()
            with open(file, 'r') as file_in:
                for line in file_in:
                    irrep_from, hypers_to = line[:-1].split('\t')
                    hypers_to = [aa.split(' x ') for aa in hypers_to.split(' + ')]
                    try:
                        hypers_to = [[tuple(irreps[1:-1].split(',')), float(n)] for n, irreps in hypers_to]
                    except:
                        print(group_from, groups_to, hypers_to)
                    rule_data[irrep_from] = hypers_to
            
            # Save to dictionary
            if group_from in self.branching_rules:
                self.branching_rules[group_from].append([groups_to, rule_data])
            else:
                self.branching_rules[group_from] = [[groups_to, rule_data]]

        total_rules = sum([len(aa) for aa in self.branching_rules.values()])
        print(f'DONE. Loaded a total of {total_rules} branching rules for {len(self.branching_rules)} simple groups.\n')

    def build_higgs_DAG(self):
        """Creates a (D)irected (A)cyclic (G)raph representing possible Higgsing patterns."""

        # Higgs all models, storing parent/child relationships
        for model in tqdm(self.models, ascii=True, desc='  Models'):
            model.children = self.higgs_hypers(model.hypers)
            for mod in model.children:
                mod.parents.add(model)

        # Share children amongst triality-equivalent models
        for model in self.models:
            for mod in model.triality_equivalent:
                mod.children.update(model.children)

        # Scan through setting reachable sinks
        for model in self.models:
            if model.is_sink():
                model.reachable_sinks = set([model])
            else:
                model.reachable_sinks = set()
                for child in model.children:
                    model.reachable_sinks.update(child.reachable_sinks)

    def higgs_hypers(self, hypers):
        """Finds possible Higgsing patterns by applying branching rules to the model with given hypers."""

        # Record group IDs and hypers when restricted to each vertex individually
        group_IDs = [irrep[4] + irrep.split('-')[1][1:].rjust(2, '0') for irrep in list(hypers.keys())[0]]
        hypers_by_vertex = [restrict_hypers(hypers, [ii]) for ii in range(len(group_IDs))]

        # Identify branching rules which work on hypers when restricted to a single vertex (None for no branching)
        # num_increased_by_vertex keeps track of how the number of vertices changes under each branching rule
        # (since, for example, for Sp(8) → Sp(4)xSp(4) the index positions of the vertices will shift by one)
        branching_rules_by_vertex = [[None] for _ in range(len(group_IDs))]
        num_increased_by_vertex = [[0] for _ in range(len(group_IDs))]
        
        for ii, group_ID in enumerate(group_IDs):
            if group_ID in self.branching_rules:
                for groups_to, branching_rule in self.branching_rules[group_ID]:

                    hypers_branched = apply_branching_rule(hypers_by_vertex[ii], branching_rule, 0)

                    # Remove su2/su3/sp2 factors
                    hypers_branched = remove_SU2_SU3_Sp2(hypers_branched)
                    if hypers_branched is None:
                        continue

                    # Save as a viable branching rule if multiplicities of charged hypers
                    # for this vertex are non-negative
                    charged_multiplicity_nonneg = all([n > 0 or all([irrep_H(irrep) == 1 for irrep in irreps])
                                                        for irreps, n in hypers_branched.items()])
                    if charged_multiplicity_nonneg:
                        branching_rules_by_vertex[ii].append(branching_rule)
                        num_increased_by_vertex[ii].append(len(groups_to) - 1)


        # Now to find combinations of branching rules. If a single branching rule works
        # when applied to the model (i.e. is compatible with (2+)-charged hypers, not just
        # when restricting to the vertex in question), then there is no need to consider
        # it in combination with branchings for other vertices. However, if a branching rule
        # which is successful when restricting to the single vertex doesn't work on the full
        # model (because hypers are tied up in (2+)-charged hypers), then it should be tried
        # in combination with others. This continues iteratively, trying branchings for k vertices
        # for which no subset of (k-1) was successful.
        
        # Identify sets of vertices which are identical
        equivalent_sets = []
        for ii in range(len(group_IDs)):
            added = False
            for subset in equivalent_sets:
                swap_permutation = list(range(len(group_IDs)))
                swap_permutation[ii] = subset[0]
                swap_permutation[subset[0]] = ii
                hypers_swapped = permute_hypers(hypers, swap_permutation)
                if compare_hypers(hypers, hypers_swapped) == 'eq':
                    subset.append(ii)
                    added = True
                    break
            if not added:
                equivalent_sets.append([ii])
        # Remove singletons
        equivalent_sets = [subset for subset in equivalent_sets if len(subset) > 1]


        # Sets of indices specifying the branching rule combinations are in br_candidates
        # (index 0 is always 'None', i.e. no branching). These are built up recursively and
        # candidates are discarded if when restricting to the vertices which are already branched
        # results in a negative multiplicity.
        br_candidates = [[0 for _ in range(len(group_IDs))]]
        hypers_higgsed = []
        successful_indices = []

        for _ in range(len(group_IDs)):
            br_candidates_next = []

            # Loop over branching rule sets with k non-trivial branchings
            for br_indices in br_candidates:
                # Identify the largest ii for which a non-trivial branching rule for vertex_ii
                # is specified by br_indices (i.e. has index > 0)
                ii_max_set = -1 if max(br_indices) == 0 else np.where(np.array(br_indices) > 0)[0][-1]

                # Build a list of branching rule sets with (k+1) non-trivial branchings
                # by selecting a non-trivial branching rule index for one of the vertices
                # at larger ii than the maximum already set
                br_indices_list = []
                for ii in range(ii_max_set+1, len(group_IDs)):
                    for index in range(1, len(branching_rules_by_vertex[ii])):
                        # Set the branching rule index for this new vertex
                        indices = br_indices.copy()
                        indices[ii] = index

                        # Check indices for equivalent sets are weakly decreasing
                        equiv_sets_okay = True
                        for subset in equivalent_sets:
                            for jj_1, jj_2 in zip(subset[:-1], subset[1:]):
                                if indices[jj_1] < indices[jj_2]:
                                    equiv_sets_okay = False
                                    break
                        if not equiv_sets_okay:
                            break

                        # Check that removing any one of branching rules always results
                        # in a branching rule set of k non-trivial branchings that is
                        # in the list of candidates
                        no_subsets_successful = True
                        for jj in range(len(group_IDs)):
                            if indices[jj] == 0:
                                continue
                            indices_less_one = indices.copy()
                            indices_less_one[jj] = 0

                            # Sort indices for equivalent sets to be decreasing
                            indices_less_one = np.array(indices_less_one)
                            for subset in equivalent_sets:
                                sorted_values = sorted(indices_less_one[subset])[::-1]
                                indices_less_one[subset] = sorted_values
                            indices_less_one = list(indices_less_one)

                            if indices_less_one in successful_indices:
                                no_subsets_successful = False
                                break

                        if no_subsets_successful:
                            br_indices_list.append(indices)

                # Loop over all newly-created branching rule sets of size (k+1)
                for br_indices_new in br_indices_list:
                    # Apply branching rules, working backwards to allow
                    # for the number of vertices to grow (e.g. from Sp(8) → Sp(4) x Sp(4))
                    hypers_new = hypers
                    for ii in range(len(br_indices_new)-1, -1, -1):
                        br_ii = branching_rules_by_vertex[ii][br_indices_new[ii]]
                        if br_ii is not None:
                            hypers_new = apply_branching_rule(hypers_new, br_ii, ii)
                        if hypers_new is None:
                            break
                    if hypers_new is None:
                        continue
                    
                    # Largest index for initial vertices for which a non-trivial branching rule
                    # has been chosen. Any future branching only changes vertices with ii > ii_max
                    ii_max_set_new = np.where(np.array(br_indices_new) > 0)[0][-1]

                    # Remove su2/su3/sp2 factors
                    hypers_removed = remove_SU2_SU3_Sp2(hypers_new)
                    if hypers_removed is None:
                        continue

                    # Check if nontrivial (at least one index > 0) and save
                    # to higgsed hypers if fully successful. Since br_indices_new
                    # is never added to br_candidates_next, no branchings which included this
                    # as a subset will be considered in the future
                    if max(br_indices_new) > 0 and all([n > 0 for n in hypers_removed.values()]):
                        hypers_higgsed.append(hypers_removed)
                        # successful_indices.append(br_indices_new)
                        continue

                    # However, since branching may increase the number of vertices and then
                    # removing su2/su3/sp2 may reduces them, we have to figure out where the
                    # initial vertices of indices ii > ii_max end up after these changes

                    # Extract how much each applied branching rule increases the number of vertices by
                    num_increased = [num_increased_by_vertex[ii][index] for ii, index in enumerate(br_indices_new)]
                    num_increased = sum(num_increased)

                    # Count the number of simple factors which were removed from hypers_new
                    # when computing hypers_removed
                    num_removed = len([irrep for irrep in list(hypers_new.keys())[0]
                                                if irrep.split('-')[1] in ['A1', 'A2', 'C2']])

                    # Set of indices for hypers_removed which are not affected by future branchings
                    ii_fixed = list(range(ii_max_set_new + num_increased - num_removed + 1))

                    # Restrict the hypers to those vertices which no future branching rules will affect
                    # The resulting charged hyper multiplicities must be non-negative
                    hypers_removed_restricted = restrict_hypers(hypers_removed, ii_fixed)
                    charged_multiplicity_nonneg = all([n > 0 or all([irrep_H(irrep) == 1 for irrep in irreps])
                                                        for irreps, n in hypers_removed_restricted.items()])
                    
                    if charged_multiplicity_nonneg:
                        br_candidates_next.append(br_indices_new)

            br_candidates = br_candidates_next


        # Finally, for all sets of hypers resulting from a successful Higgsing, find the corresponding model
        higgsed_models = set()
        for hypers_new in hypers_higgsed:
            model = self.find_model(hypers_new)
            if model is not None:
                higgsed_models.add(model)

        return higgsed_models

    def find_triality_equivalent(self, model):
        """Finds all models which are related by a triality transformation of one or more SO(8) factors."""

        # Identify vertices which are individually triality-invariant
        ii_tri = [ii for ii, vertex_ID in enumerate(model.vertex_IDs)
                    if vertex_ID in self.triality_invariant_vertex_IDs]
        if len(ii_tri) == 0:
            return set()

        hypers_transformed = [model.hypers]

        for ii in ii_tri:
            hypers_transformed_new = [apply_triality_transformation(hypers, ii) for hypers in hypers_transformed]
            hypers_transformed_new = [aa for bb in hypers_transformed_new for aa in bb]
            hypers_transformed.extend(hypers_transformed_new)

        # Find models with these hypers
        triality_equivalent_models = set()
        for hypers in hypers_transformed:
            model_tri_equiv = self.find_model(hypers)
            if model_tri_equiv is not None:
                triality_equivalent_models.add(model_tri_equiv)

        # Remove the starting model itself
        triality_equivalent_models.discard(model)

        return triality_equivalent_models

    def find_model(self, hypers):
        """Finds the model with the given hypers using `hyper_hash`."""

        hypers_canonical = to_canonical_order(hypers)
        hashed = hyper_hash(hypers_canonical)

        if hashed in self.models_by_hash:
            return self.models_by_hash[hashed]
        
        print('Model with the following hypers (in canonical order) not found:')
        display_hypers(hypers_canonical)
        print('hash:', hashed)
        return None

class Model():

    def __init__(self, T, tsv_string=None):
        
        self.T = T

        if tsv_string is None:
            # Initialize to "empty" model with V=0
            self.ID = 'mdl-(0,0,0)-0-EMPTY'
            self.kS, self.kN, self.kT, self.k = 0, 0, 0, 0

            self.delta = 0

            self.vertex_IDs = []
            self.gauge_group = []
            self.gauge_group_rank = 0
            self.gauge_group_dim = 0

            if self.T == 0:
                self.bI_options = {'odd': [np.array([[3]])]}
            elif self.T == 1:
                self.bI_options = {'odd': [np.array([[3, 1]])], 'even': [np.array([[2, 2]])]}

            self.gram_bIbJ = np.array([9-self.T], dtype=int)
            self.hypers = {tuple(): 273 - 29*self.T}
            self.hash = None

            self.parents = set()
            self.children = set()
            self.reachable_sinks = set()

        else:
            self.initialize_from_tsv(tsv_string)
        
    def initialize_from_tsv(self, tsv_string):
        """Initializes a model from a .tsv file."""

        # Remove newline character and split by tabs
        self.tsv_string = tsv_string
        data = tsv_string[:-1].split('\t')

        # Model ID and number of vertices of each type
        self.ID = data[0]
        self.kS, self.kN, self.kT = [int(xx) for xx in data[0].split('(')[1].split(')')[0].split(',')]
        self.k = self.kS + self.kN + self.kT

        # Vertex info
        self.vertex_IDs = data[1][12:-1].split(', ')

        # Gauge group info
        self.gauge_group = []
        self.gauge_group_rank = 0
        self.gauge_group_dim = 0
        for vertex_ID in self.vertex_IDs:
            group_type = vertex_ID[4]
            group_rank = int(vertex_ID[5:7])
            self.gauge_group_rank += group_rank

            if group_type == 'A':
                self.gauge_group.append(f'SU({group_rank+1})')
                self.gauge_group_dim += group_rank * (group_rank + 2)
            elif group_type == 'B':
                self.gauge_group.append(f'SO({2*group_rank+1})')
                self.gauge_group_dim += group_rank * (2*group_rank + 1)
            elif group_type == 'C':
                self.gauge_group.append(f'Sp({group_rank})')
                self.gauge_group_dim += group_rank * (2*group_rank + 1)
            elif group_type == 'D':
                self.gauge_group.append(f'SO({2*group_rank})')
                self.gauge_group_dim += group_rank * (2*group_rank - 1)
            elif group_type == 'E':
                self.gauge_group.append(f'E({group_rank})')
                self.gauge_group_dim += 78 if group_rank == 6 else 133 if group_rank == 7 else 248
            elif group_type == 'F':
                self.gauge_group.append(f'F({group_rank})')
                self.gauge_group_dim += 52
            elif group_type == 'G':
                self.gauge_group.append(f'G({group_rank})')
                self.gauge_group_dim += 14

        self.delta = int(data[2][4:])

        # Lattice info
        self.bI_options = dict()
        bI_odd_data = data[3][10:-1]
        bI_even_data = data[4][11:-1]

        if len(bI_odd_data) > 0:
            self.bI_options['odd'] = []
            for bI_arr in bI_odd_data[2:-2].split(']], [['):
                bI_arr = np.array([[int(xx) for xx in bI.split(',')] for bI in bI_arr.split('],[')])
                self.bI_options['odd'].append(bI_arr)

                if len(bI_arr[0]) == 1:
                    # For T=0 where bI are just integers
                    self.gram_bIbJ = bI_arr @ bI_arr.T
                else:
                    # For T=1 where bI are vectors
                    self.gram_bIbJ = bI_arr @ OMEGA_ODD @ bI_arr.T

        if len(bI_even_data) > 0:
            self.bI_options['even'] = []
            for bI_arr in bI_even_data[2:-2].split(']], [['):
                bI_arr = np.array([[int(xx) for xx in bI.split(',')] for bI in bI_arr.split('],[')])
                self.bI_options['even'].append(bI_arr)
                
                self.gram_bIbJ = bI_arr @ OMEGA_EVEN @ bI_arr.T
        
        # Hypermultiplets
        self.hypers = dict()

        # Add in required number of neutral hypers
        irreps_trivial = [f'irr-{v_ID[4]}{int(v_ID[5:7])}-1' for v_ID in self.vertex_IDs]
        if self.delta != 273 - 29*self.T:
            self.hypers[tuple(irreps_trivial)] = 273 - 29*self.T - self.delta

        for hyperstring in data[5:]:
            indices, hypers = hyperstring.split(' : ')
            indices = tuple(int(xx) for xx in indices[1:-1].split(','))
            hypers = [xx.split(' x ') for xx in hypers.split(' + ')]

            for n, irreps in hypers:
                irreps_all = irreps_trivial.copy()
                for ii, irrep in zip(indices, irreps[1:-1].split(', ')):
                    irreps_all[ii-1] = irrep

                nR = float(n)
                if nR % 1 == 0:
                    nR = int(nR)
                self.hypers[tuple(irreps_all)] = nR

        self.hash = None

        self.parents = set()
        self.children = set()
        self.reachable_sinks = set()

    def display(self):
        
        print('\n' + self.ID)
        print(len(self.ID)*'—')
        print(f'Δ = {self.delta}')
        print('vertices:', '[' + ', '.join(self.vertex_IDs) + ']')
        print('bI.bJ =')
        print(self.gram_bIbJ)

        if 'odd' in self.bI_options:
            print('bI-odd =')
            for bI in self.bI_options['odd']:
                print(bI.T)
                
        if 'even' in self.bI_options:
            print('bI-even =')
            for bI in self.bI_options['even']:
                print(bI.T)

        print('hypers =')
        display_hypers(self.hypers)

    def get_hyper_hash(self):
        """Returns hash of hypermultiplets after bringing to a canonical order."""
        if self.hash is None:
            hypers_ordered = to_canonical_order(self.hypers)
            self.hash = hyper_hash(hypers_ordered)
        return self.hash

    def is_sink(self):
        """Returns whether the model has any children in the Higgs DAG."""
        return len(self.children) == 0
    
    def is_source(self):
        """Returns whether the model has any parents in the Higgs DAG."""
        return len(self.parents) == 0

class Vertex():

    def __init__(self, tsv_string):
        # Remove '\n' and split by tabs
        data = tsv_string[:-1].split('\t')

        self.ID = data[0]
        self.delta = int(data[1][4:])
        self.bibi = int(data[2][8:])
        self.b0bi = int(data[3][8:])

        # Build hypers dictionary
        if len(data[4]) == 9:
            self.hypers = dict()
        else:
            hypers = [hyp.split(' x ') for hyp in data[4][9:].split(' + ')]
            self.hypers = {irr[1:-1]: float(nR) for nR, irr in hypers}
            self.hypers = {irr: int(nR) if nR % 1 == 0 else nR for irr, nR in self.hypers.items()}

        # Group info
        self.group_rank = int(self.ID[5:7])
        if self.ID[4] == 'A':
            self.group_type = 'SU'
            self.group_N = self.group_rank + 1
        elif self.ID[4] == 'B':
            self.group_type = 'SO'
            self.group_N = 2*self.group_rank + 1
        elif self.ID[4] == 'C':
            self.group_type = 'Sp'
            self.group_N = self.group_rank
        elif self.ID[4] == 'D':
            self.group_type = 'SO'
            self.group_N = 2*self.group_rank
        else:
            self.group_type = 'EFG'
            self.group_N = self.group_rank

    def display(self):
        print(f'{self.ID:24}    Δ = {self.delta:4}    bi.bi = {self.bibi:4}    b0.bi = {self.b0bi:4}    hypers = ', end='')
        print(' + '.join([f'{n} x {irr}' for irr, n in self.hypers.items()]))

    def is_triality_invariant(self):
        """Determines whether the hypers are invariant under triality transformation.
        
        If the gauge algebra is not so(8) or if there are no hypers, returns False.

        In this ensemble the only relevant irreps of so(8) are those of dimension 8, 35, 56 and 112.
        Since conjugate irreps are treated collectively, this means the number of 8s needs
        to be twice that of 8v (so that n(8v) = n(8s) = n(8c) after assigning half of the 8s to be 8c)
        and similarly for the others.
        """

        if self.group_type != 'SO' or self.group_N != 8 or len(self.hypers) == 0:
            return False
        
        num_8_v,   num_8_s   = 0, 0
        num_35_v,  num_35_s  = 0, 0
        num_56_v,  num_56_s  = 0, 0
        num_112_v, num_112_s = 0, 0

        # Get number of each irrep from hypers if nonzero
        if 'irr-D4-8a'   in self.hypers: num_8_v   = self.hypers['irr-D4-8a']
        if 'irr-D4-8b'   in self.hypers: num_8_s   = self.hypers['irr-D4-8b']
        if 'irr-D4-35a'  in self.hypers: num_35_v  = self.hypers['irr-D4-35a']
        if 'irr-D4-35b'  in self.hypers: num_35_s  = self.hypers['irr-D4-35b']
        if 'irr-D4-56a'  in self.hypers: num_56_v  = self.hypers['irr-D4-56a']
        if 'irr-D4-56b'  in self.hypers: num_56_s  = self.hypers['irr-D4-56b']
        if 'irr-D4-112a' in self.hypers: num_112_v = self.hypers['irr-D4-112a']
        if 'irr-D4-112b' in self.hypers: num_112_s = self.hypers['irr-D4-112b']

        return (2*num_8_v == num_8_s) and (2*num_35_v == num_35_s) \
                and (2*num_56_v == num_56_s) and (2*num_112_v == num_112_s)

def display_hypers(hypers):
    """Formats and prints hypers dict."""

    if hypers is None:
        print('None')
        return

    # Collect longest irrep ID strings for each column
    nchars = np.max([
        [1 if irrep[-2:] == '-1' else len(irrep) for irrep in irreps] for irreps in hypers
    ], axis=0)

    for irreps in get_sorted_hyper_keys(hypers):
        # Replace trivial reps with bullet for clarity, then pad to constant widths
        irreps_padded = ['•' if irrep[-2:] == '-1' else irrep for irrep in irreps]
        irreps_padded = [irrep.ljust(nchar) for nchar, irrep in zip(nchars, irreps_padded)]

        # Irrep multiplicity (only show if ≠1)
        if hypers[irreps] != 1:
            print(f'{hypers[irreps]:4} × ', end='')
        else:
            print(end=7*' ')
        
        # Irrep itself
        print('(' + ' '.join(irreps_padded) + ')')
    print()

def get_sorted_hyper_keys(hypers):
    """Returns keys for hypers dict in canonical order."""

    if len(hypers) == 1:
        return hypers.keys()

    hyper_data = np.array([
        [
            *sorted([ii if irrep[-2:] != '-1' else -1 for ii, irrep in enumerate(irreps)]),
            *[irrep_H(irrep) for irrep in irreps],
            *irreps
        ] for irreps in hypers
    ], dtype=object)

    order = np.lexsort(hyper_data.T[::-1])
    keys_sorted = list(hypers.keys())
    keys_sorted = [keys_sorted[ii] for ii in order]

    return keys_sorted

def irrep_H(irrep):
    """Returns value of H for the irrep (=dim/2 for quaternionic, =dim otherwise)."""

    # Extract irrep dimension from irrep ID
    dim = int(''.join([aa for aa in irrep.split('-')[2] if aa.isnumeric()]))

    # For half-hypers, H = dim/2
    if irrep[-2:] == '-h':
        return dim // 2
    else:
        return dim

def to_canonical_order(hypers):
    """Sorts vertices (and corresponding hypers) to bring to a canonical form.

    This is an instance of the (labelled) graph isomorphism problem,
    which is notoriously difficult. However, the models we encounter
    remain relatively small so this does not become an issue.

    The canonical order is specified in the following way:
        i) vertices are ordered with IDs lexicographically (weakly) increasing
        ii) amongst permutations satisfying (i), the order is determined
            by the total order on hypers (defined through compare_hypers())
    Notice that for a model with all distinct vertices, condition (i)
    already uniquely determines the canonical order.
    """

    # If the model has symmetries then the "largest" hypers (as determined by total ordering)
    # may be achieved by many permutations of the vertices (e.g. if a k-model has k identical
    # vertices and k(k-1)/2 identical edges then all k! permutations are equivalent). We can
    # avoid checking a large number of permutations by first identifying symmetries of the
    # model, i.e. identifying groups of vertices which can all be swapped pair-wise with no
    # effect on the hypers.

    # Number of vertices (i.e. number of irreps in each key)
    n_vertices = len(list(hypers.keys())[0])

    # Run through hypers tallying total H for each vertex
    H_totals = n_vertices * [0]

    for irreps, n in hypers.items():
        Hs = [irrep_H(irrep) for irrep in irreps]
        H_tot = np.prod(Hs)
        for ii, H_ii in enumerate(Hs):
            if H_ii != 1:
                H_totals[ii] += n * H_tot

    # Identify groups of vertices which can be interchanged pair-wise
    # without changing the hypers (these correspond to symmetries of the model
    # and wlg the vertices in these groups can be taken to be in some fixed order)
    identical_vertices = []
    for ii, H_total_ii in enumerate(H_totals):
        added = False
        for subgroup in identical_vertices:
            jj = subgroup[0]
            if H_total_ii != H_totals[jj]:
                continue

            # Swap ii with jj and see if hypers unchanged
            permutation = list(range(n_vertices))
            permutation[ii] = jj
            permutation[jj] = ii

            hypers_permuted = permute_hypers(hypers, permutation)
            if compare_hypers(hypers, hypers_permuted) == 'eq':
                subgroup.append(ii)
                added = True
                break
        if not added:
            identical_vertices.append([ii])

    # Now build potential permutations by successively adding in subsets
    permutation_candidates = [[]]
    for subset in identical_vertices:
        H_total_subset = H_totals[subset[0]]
        permutation_candidates_next = []
        for permutation in permutation_candidates:
            for ii in range(len(permutation) + 1):
                if ii > 0 and H_totals[permutation[ii-1][0]] > H_total_subset:
                    continue
                if ii < len(permutation) and H_totals[permutation[ii][0]] < H_total_subset:
                    continue
                permutation_new = permutation.copy()
                permutation_new.insert(ii, subset)
                permutation_candidates_next.append(permutation_new)
        permutation_candidates = permutation_candidates_next
    permutation_candidates = [[aa for bb in permutation for aa in bb] for permutation in permutation_candidates]

    # Find canonical hypers by comparing all candidate permutations
    permutation_canonical = None
    hypers_canonical = None
    for permutation in permutation_candidates:
        hypers_permuted = permute_hypers(hypers, permutation)
        if permutation_canonical is None or compare_hypers(hypers_permuted, hypers_canonical) == 'gt':
            permutation_canonical = permutation
            hypers_canonical = hypers_permuted

    return hypers_canonical

def compare_hypers(hypers_1, hypers_2):
    """Defines a total order on collections of hypers.
    
    Returns 'lt' if hypers_1 < hypers_2, 'gt' if hypers_1 > hypers_2 and 'eq' if hypers_1 = hypers_2.
    """

    # First check numbers of irreps: if unequal, this determines the order
    if len(hypers_1) != len(hypers_2):
        return 'lt' if len(hypers_1) < len(hypers_2) else 'gt'
    
    # Same number of irreps: scan through in a canonical order determined by 'get_sorted_hyper_keys'
    for irreps_1, irreps_2 in zip(get_sorted_hyper_keys(hypers_1), get_sorted_hyper_keys(hypers_2)):
        if irreps_1 != irreps_2:
            # Irreps are different: order determined by Irrep.__lt__()
            return 'lt' if irreps_1 < irreps_2 else 'gt'
        if hypers_1[irreps_1] != hypers_2[irreps_2]:
            # Same irrep but different number
            return 'lt' if hypers_1[irreps_1] < hypers_2[irreps_2] else 'gt'
        
    return 'eq'

def permute_hypers(hypers, permutation):
    """Return hypers after reordering vertices/gauge factors according to the given permutation."""
    # For example,
    #   hypers = {('irr-B3-7', 'irr-D4-8a', 'irr-G2-14'): 3, ...}
    #   permutation = [2, 0, 1]
    #   permute_hypers(hypers, permutation)
    #   >> {('irr-G2-14', 'irr-B3-7', 'irr-D4-8a'): 3, ...}
    return {tuple(irreps[ii] for ii in permutation): n for irreps, n in hypers.items()}

def restrict_hypers(hypers, indices):
    """Restricts hypers to irreps of the subgroup at given indices."""

    hypers_restricted = dict()

    for irreps, n in hypers.items():
        irreps_restricted = tuple(irrep for ii, irrep in enumerate(irreps) if ii in indices)
        n_restricted = n * int(np.prod([irrep_H(irrep) for ii, irrep in enumerate(irreps) if ii not in indices]))
        add_to_dict(hypers_restricted, irreps_restricted, n_restricted)

    group_IDs = np.array([irrep.split('-')[1] for irrep in list(hypers.keys())[0]])
    triv_irrep = tuple(trivial_irrep(g_ID) for ii, g_ID in enumerate(group_IDs) if ii in indices)
    n_to_remove = sum([irrep_H(adjoint_irrep(g_ID)) for ii, g_ID in enumerate(group_IDs) if ii not in indices])
    add_to_dict(hypers_restricted, triv_irrep, -n_to_remove)

    return hypers_restricted

def hyper_hash(hypers):
    """Hash function for hypers."""

    # Use canonical order for irreps defined by 'get_sorted_hyper_keys' and join data into string
    keys = get_sorted_hyper_keys(hypers)
    hypers_string = ' + '.join([f'{hypers[key]} x (' + ','.join(key) + ')' for key in keys])

    # Use sha1() for reproducability
    # Python's built-in hash() depends on a random seed
    return int(sha1(hypers_string.encode()).hexdigest(), 16)

def remove_SU2_SU3_Sp2(hypers):
    """Completely Higgses all possible SU(2/3), Sp(2) factors."""

    if hypers is None:
        return None

    group_IDs = np.array([irrep.split('-')[1] for irrep in list(hypers.keys())[0]])

    ii_SU2 = [ii for ii, g_ID in enumerate(group_IDs) if g_ID == 'A1']
    ii_SU3 = [ii for ii, g_ID in enumerate(group_IDs) if g_ID == 'A2']
    ii_Sp2 = [ii for ii, g_ID in enumerate(group_IDs) if g_ID == 'C2']
    ii_to_remove = [*ii_SU2, *ii_SU3, *ii_Sp2]

    if len(ii_to_remove) == 0:
        return hypers
    
    # Check that Δ is positive for each factor being removing
    # and that there is a hyper with positive multiplicity
    # for that factor that is *only* charged under factors being removed
    Δs = [-irrep_H(adjoint_irrep(g_ID)) for g_ID in group_IDs]
    hyp_nonneg = [False for _ in range(len(group_IDs))]
    neutrals_before = 0

    for irreps, n in hypers.items():
        Hs = [irrep_H(irrep) for irrep in irreps]
        H_total = np.prod(Hs)
        charged_under = [ii for ii, H_ii in enumerate(Hs) if H_ii > 1]
        charged_under_kept_factors = len([ii for ii in charged_under if ii not in ii_to_remove]) > 0

        if max(Hs) == 1:
            neutrals_before = n

        for ii in ii_to_remove:
            if Hs[ii] > 1:
                Δs[ii] += n*H_total
                if not charged_under_kept_factors:
                    hyp_nonneg[ii] = True

    for ii in ii_to_remove:
        if Δs[ii] < 0 or not hyp_nonneg[ii]:
            return None

    # Form resulting hypers from deleting SU(2/3), Sp(2) factors
    hypers_stripped = dict()
    for irreps, n in hypers.items():
        irreps_new = tuple(irrep for ii, irrep in enumerate(irreps)
                            if ii not in ii_SU2 and ii not in ii_SU3 and ii not in ii_Sp2)
        n_new = n * np.prod([irrep_H(irrep) for ii, irrep in enumerate(irreps)
                                if ii in ii_SU2 or ii in ii_SU3 or ii in ii_Sp2])
        add_to_dict(hypers_stripped, irreps_new, n_new)

    irrep_triv = tuple(trivial_irrep(g_ID) for g_ID in group_IDs if g_ID not in ['A1', 'A2', 'C2'])
    add_to_dict(hypers_stripped, irrep_triv, -3*len(ii_SU2) - 8*len(ii_SU3) - 10*len(ii_Sp2))
    hypers_stripped = {irreps: int(n) if n % 1 == 0 else n for irreps, n in hypers_stripped.items() if n != 0}

    neutrals_after = hypers_stripped.get(irrep_triv, 0)

    if neutrals_after < neutrals_before:
        return None

    return hypers_stripped

def apply_branching_rule(hypers: dict, branching_rule: dict, ii: int):
    """Applies branching rule to vertex at index ii."""

    if hypers is None:
        return None
    
    group_IDs = [irrep.split('-')[1] for irrep in list(hypers.keys())[0]]
    target_group_IDs = [irrep.split('-')[1] for irrep in list(branching_rule.values())[0][0][0]]

    hypers_branched = dict()

    # Branch hypers
    for irreps, n in hypers.items():
        if irreps[ii] not in branching_rule:
            print(f'Abandoning branching: rule is missing for {irreps[ii]}')
            return None

        for irreps_branched, n_branched in branching_rule[irreps[ii]]:
            irreps_new = tuple([*irreps[:ii], *irreps_branched, *irreps[(ii+1):]])
            add_to_dict(hypers_branched, irreps_new, n * n_branched)

    # Subtract off branched adjoint
    adj_ii = adjoint_irrep(group_IDs[ii])
    irreps_trivial = [trivial_irrep(g_ID) for g_ID in group_IDs]
    for irreps_branched, n_branched in branching_rule[adj_ii]:
        irreps_new = tuple([*irreps_trivial[:ii], *irreps_branched, *irreps_trivial[(ii+1):]])
        add_to_dict(hypers_branched, irreps_new, -n_branched)

    # Add in adjoint for target group
    group_IDs_branched = [*group_IDs[:ii], *target_group_IDs, *group_IDs[(ii+1):]]
    irreps_trivial_branched = [trivial_irrep(g_ID) for g_ID in group_IDs_branched]
    for jj, g_ID_jj in enumerate(target_group_IDs):
        irreps_new = irreps_trivial_branched.copy()
        irreps_new[ii+jj] = adjoint_irrep(g_ID_jj)
        irreps_new = tuple(irreps_new)
        add_to_dict(hypers_branched, irreps_new, 1)

    # Make multiplicities integers and remove irreps with nR = 0
    hypers_branched = {irreps: int(n) if n % 1 == 0 else n for irreps, n in hypers_branched.items() if n != 0}

    return hypers_branched

def apply_triality_transformation(hypers, ii):
    """Apply all possible triality transformations to the triality-invariant vertex at index ii.
    
    Transformations only differ in how they affect multi-charged hypers.
    """

    # For each trio of triality-related irreps, assign half of spinor irreps
    # to be conjugate irreps in all possible ways
    hypers_w_conjugates = [hypers]
    for D4_irr_family in ['irr-D4-8', 'irr-D4-35', 'irr-D4-56', 'irr-D4-112']:
        hypers_w_conjugates = [triality_assign_conjugate_irreps(hyps, ii, D4_irr_family)
                                for hyps in hypers_w_conjugates]
        hypers_w_conjugates = [aa for bb in hypers_w_conjugates for aa in bb]

    # Then apply triality transformation:
    #   v → s
    #   s → c (→ s)
    #   c → v
    hypers_transformed = []
    for hypers_w_conj in hypers_w_conjugates:
        hypers_new = dict()
        for irreps, n in hypers_w_conj.items():
            irreps_triality = list(irreps)
            if   irreps_triality[ii][-1] == 'a': irreps_triality[ii] = irreps_triality[ii][:-1] + 'b'
            if   irreps_triality[ii][-1] == 'c': irreps_triality[ii] = irreps_triality[ii][:-1] + 'a'
            irreps_triality = tuple(irreps_triality)
            add_to_dict(hypers_new, irreps_triality, n)
        hypers_new = {irreps: int(n) if n % 1 == 0 else n for irreps, n in hypers_new.items()}
        hypers_transformed.append(hypers_new)

    return hypers_transformed

def triality_assign_conjugate_irreps(hypers, ii, D4_irr_family):
    """Interpret s-type irreps of so(8) at index ii as combinations of s/c irreps in all possible ways."""

    if D4_irr_family not in ['irr-D4-8', 'irr-D4-35', 'irr-D4-56', 'irr-D4-112']:
        raise Exception('Triality transformations only implemented for irr-D4-8, irr-D4-35, irr-D4-56, irr-D4-112')

    # For each hyper, identify number of s-type irreps
    total_spinors = 0
    num_spinor = []
    n_max = []
    half_hypers = []
    for irreps, n in hypers.items():
        if irreps[ii] != D4_irr_family + 'b':
            num_spinor.append(0)
            n_max.append(0)
            half_hypers.append('NA')
            continue
        H_others = int(np.prod([irrep_H(irrep) for jj, irrep in enumerate(irreps) if jj != ii]))
        total_spinors += int(round(n*H_others))

        if any([irrep[-2:] == '-q' for irrep in irreps]):
            n_max.append(int(2*n))
            num_spinor.append(H_others//2)
            half_hypers.append(True)
        else:
            n_max.append(n)
            num_spinor.append(H_others)
            half_hypers.append(False)

    spinor_goal = total_spinors // 2

    # Exactly half of total spinors must be assigned to be conjugate spinors
    n_conjugate = [np.zeros(len(n_max), dtype=int)]

    for jj, n_max_jj in enumerate(n_max):
        n_conjugate_new = []
        for n_config in n_conjugate:
            for n_jj in range(n_max_jj+1):
                n_config_new = n_config.copy()
                n_config_new[jj] = n_jj
                if n_config_new @ num_spinor > spinor_goal:
                    break
                n_conjugate_new.append(n_config_new)
        n_conjugate = n_conjugate_new

    n_conjugate = [n_config for n_config in n_conjugate if n_config @ num_spinor == spinor_goal]

    # Include also the untransformed hypers
    hypers_transformed = []
    for n_config in n_conjugate:
        hypers_w_conj = dict()
        for n_to_conj, half_hyper, (irreps, n) in zip(n_config, half_hypers, hypers.items()):
            if n_to_conj == 0:
                add_to_dict(hypers_w_conj, irreps, n)
                continue

            # Change to conjugate
            irreps_conj = list(irreps)
            irreps_conj[ii] = D4_irr_family + 'c'
            irreps_conj = tuple(irreps_conj)

            n_transformed = n_to_conj/2 if half_hyper else n_to_conj
            n_remain = n - n_transformed
            if n_transformed % 1 == 0: n_transformed = int(n_transformed)
            if n_remain % 1 == 0: n_remain = int(n_remain)

            add_to_dict(hypers_w_conj, irreps_conj, n_transformed)
            add_to_dict(hypers_w_conj, irreps,  n_remain)

        hypers_transformed.append(hypers_w_conj)
        
    return hypers_transformed

def add_to_dict(dictionary, key, value):
    """Adds value to a dictionary.
    
    If key is already in dictionary, *adds* value (rather than overwriting it).
    If key if not present, adds "key: value" pair to the dictionary.
    """

    if value == 0:
        return

    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value

def adjoint_irrep(group_ID):
    """Returns irrep_ID for adjoint representation of given group."""

    group_type = group_ID[0]
    n = int(group_ID[1:])

    if   group_type == 'A': return f'irr-A{n}-{n*(n+2)}'
    elif group_type == 'B': return f'irr-B{n}-{n*(2*n+1)}'
    elif group_type == 'C': return f'irr-C{n}-{n*(2*n+1)}'
    elif group_type == 'D': return f'irr-D{n}-{n*(2*n-1)}'

    elif n == 6: return 'irr-E6-78'
    elif n == 7: return 'irr-E7-133'
    elif n == 8: return 'irr-E8-248'
    elif n == 4: return 'irr-F4-52'
    elif n == 2: return 'irr-G2-14'

def trivial_irrep(group_ID):
    """Returns irrep_ID for trivial representation of given group."""
    return f'irr-{group_ID[0]}{int(group_ID[1:])}-1'
