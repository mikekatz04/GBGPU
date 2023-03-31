Search.setIndex({"docnames": ["GBGPU_tutorial", "README", "index", "user/derivedwaves", "user/main", "user/utils"], "filenames": ["GBGPU_tutorial.ipynb", "README.rst", "index.rst", "user/derivedwaves.rst", "user/main.rst", "user/utils.rst"], "titles": ["GBGPU Tutorial", "gbgpu: GPU/CPU Galactic Binary Waveforms", "gbgpu: GPU/CPU Galactic Binary Waveforms", "Extending GBGPU Waveforms", "Fast GB Waveforms and Likelihoods", "GBGPU Utility Functions"], "terms": {"i": [0, 1, 2, 3, 4, 5], "gpu": [0, 3, 4], "acceler": [0, 1, 2, 3], "version": 0, "fastgb": [0, 1, 2, 3, 5], "which": [0, 1, 2, 4, 5], "ha": [0, 1, 2, 3], "been": [0, 1, 2, 3, 4], "develop": [0, 1, 2], "neil": [0, 1, 2], "cornish": [0, 1, 2], "tyson": [0, 1, 2], "littenberg": [0, 1, 2], "travi": [0, 1, 2, 3], "robson": [0, 1, 2, 3], "sta": [0, 1, 2], "babak": [0, 1, 2], "It": [0, 1, 2, 3, 4], "comput": [0, 1, 2, 3, 4, 5], "system": [0, 1, 2, 4], "observ": [0, 1, 2, 3, 4, 5], "lisa": [0, 1, 2, 4], "us": [0, 1, 2, 3, 4, 5], "fast": [0, 1, 2], "slow": [0, 1, 2, 3, 5], "type": [0, 1, 2, 3, 4, 5], "decomposit": [0, 1, 2], "For": [0, 1, 2, 4], "more": [0, 1, 2, 3, 4], "detail": [0, 1, 2], "origin": [0, 1, 2, 3], "construct": [0, 1, 2, 3], "see": [0, 1, 2, 3, 4], "arxiv": [0, 1, 2, 3], "0704": [0, 1, 2], "1808": [0, 1, 2], "The": [0, 1, 2, 3, 4, 5], "current": [0, 1, 2, 4], "code": [0, 1, 2, 3, 4], "veri": [0, 1, 2], "close": [0, 1, 2], "relat": [0, 1, 2, 3], "implement": [0, 1, 2, 4], "data": [0, 1, 2, 4], "challeng": [0, 1, 2, 4], "python": [0, 1, 2, 4], "packag": [0, 1, 2, 4], "entir": [0, 1, 2], "base": [0, 1, 2, 4], "about": [0, 1, 2], "1": [0, 1, 2, 3, 4, 5], "2": [0, 1, 2, 3, 4, 5], "speed": [0, 1, 2], "full": [0, 1, 2, 4], "much": [0, 1, 2], "simpler": [0, 1, 2], "right": [0, 1, 2], "now": [0, 1, 2], "There": [0, 1, 2], "ar": [0, 1, 2, 3, 4, 5], "also": [0, 1, 2, 4], "mani": [0, 1, 2, 4], "includ": [0, 1, 2, 3], "likelihood": [0, 1, 2], "individu": [0, 1, 2, 4], "well": [0, 1, 2, 3], "method": [0, 1, 2, 3, 4], "combin": [0, 1, 2, 4], "global": [0, 1, 2, 4, 5], "fit": [0, 1, 2, 4], "templat": [0, 1, 2, 4], "cpu": [0, 4], "agnost": [0, 1, 2, 4], "cuda": [0, 1, 2, 5], "nvidia": [0, 1, 2], "requir": [0, 1, 2, 3], "run": [0, 3, 4], "document": [0, 1], "thi": [0, 1, 2, 3, 4, 5], "wa": [0, 1, 2, 3], "design": [0, 1, 2], "todo": [0, 3, 4], "add": [0, 1, 2, 3, 4], "new": [0, 3], "number": [0, 3, 4, 5], "If": [0, 1, 2, 3, 4, 5], "you": [0, 1, 2, 3, 4], "ani": [0, 1, 2, 3, 5], "part": [0, 1, 2, 3, 5], "pleas": [0, 1, 2, 3], "cite": [0, 1, 2, 3], "its": [0, 1, 2], "zenodo": [0, 1, 2], "page": [0, 1, 2], "1806": [0, 1, 2, 3], "00500": [0, 1, 2, 3], "import": [0, 1, 2], "numpi": [0, 1, 2, 4], "np": [0, 3, 4], "matplotlib": [0, 1, 2], "pyplot": 0, "plt": 0, "inlin": 0, "from": [0, 1, 2, 3, 4, 5], "thirdbodi": [0, 3], "gbgputhirdbodi": [0, 2, 3], "constant": [0, 3], "initi": [0, 3, 4, 5], "class": [0, 2, 4], "use_gpu": [0, 2, 3, 4], "fals": [0, 3, 4], "setup": [0, 1, 2, 3], "all": [0, 3, 4, 5], "oper": [0, 3], "vector": 0, "manner": 0, "so": [0, 4], "take": [0, 3, 4], "arrai": [0, 4, 5], "paramet": [0, 3, 4, 5], "input": [0, 3, 4, 5], "3": [0, 1, 2, 3], "dt": [0, 4], "10": [0, 4], "0": [0, 1, 2, 3, 4, 5], "tob": [0, 5], "4": [0, 1, 2, 4], "year": [0, 3, 4], "point": [0, 3, 4], "none": [0, 3, 4, 5], "insid": 0, "amp": [0, 3, 4, 5], "f0": [0, 3, 4, 5], "p2": [0, 3], "n": [0, 1, 2, 3, 4, 5], "batch": [0, 4], "num_bin": [0, 2, 4], "2e": 0, "23": 0, "fdot": [0, 3, 4, 5], "7": [0, 4], "538331e": 0, "18": 0, "fddot": [0, 3, 4, 5], "phi0": [0, 4], "phase": [0, 3, 4], "iota": [0, 3, 4], "inclin": [0, 3, 4], "psi": [0, 4], "polar": [0, 4], "angl": [0, 3, 4], "lam": [0, 3, 4], "eclipt": [0, 3, 4], "longitud": [0, 3], "beta_ski": 0, "5": [0, 4], "latitud": [0, 3, 4], "amp_in": 0, "f0_in": 0, "fdot_in": 0, "fddot_in": 0, "phi0_in": 0, "iota_in": 0, "psi_in": 0, "lam_in": 0, "beta_sky_in": 0, "param": [0, 4], "run_wav": [0, 2, 3, 4], "t": [0, 3, 4, 5], "oversampl": [0, 3, 4, 5], "signal": [0, 4, 5], "first": [0, 3, 4], "A": [0, 2, 4, 5], "freq": [0, 2, 4], "print": 0, "length": [0, 4], "shape": [0, 3, 4, 5], "plot": 0, "ab": 0, "ylabel": 0, "tdi": [0, 4, 5], "channel": [0, 4, 5], "fontsiz": 0, "16": 0, "xlabel": 0, "hz": [0, 3, 4, 5], "dx": 0, "7e": 0, "xlim": 0, "512": 0, "0019993": 0, "0020007000000000002": 0, "possibl": 0, "inherit": [0, 3, 4], "special": [0, 3], "inheritgbgpu": [0, 2], "http": [0, 1, 2], "mikekatz04": [0, 1, 2], "github": [0, 1, 2], "io": 0, "html": 0, "user": [0, 4], "derivedwav": 0, "__": 0, "allow": [0, 4], "other": [0, 2, 3, 4], "effect": [0, 3, 4], "vari": 0, "need": [0, 1, 2, 3, 4], "written": 0, "when": [0, 4], "prepare_additional_arg": [0, 2, 3, 4], "special_get_n": [0, 2, 3], "shift_frequ": [0, 2, 3], "add_to_arg": [0, 2, 3], "prepar": [0, 3], "argument": [0, 3, 4], "beyond": [0, 3], "put": 0, "limit": 0, "domain": [0, 3, 4, 5], "shift": [0, 3], "transfer": [0, 3], "an": [0, 3, 4], "gb_third": 0, "a2": [0, 3], "400": 0, "varpi": [0, 3], "e2": [0, 3], "eccentr": [0, 3, 4], "period": [0, 3], "t2": [0, 3], "periapsi": 0, "passag": [0, 3], "a2_in": 0, "p2_in": 0, "varpi_in": 0, "e2_in": 0, "t2_in": 0, "6": 0, "a_third": 0, "label": 0, "No": 0, "ind": [0, 4], "8": [0, 1, 2, 4], "info_matrix": 0, "information_matrix": [0, 2, 4], "easy_central_differ": [0, 4], "ep": [0, 3, 4], "1e": [0, 3, 4], "9": [0, 1, 2, 4], "1024": [0, 4], "cov": 0, "linalg": 0, "pinv": 0, "covari": [0, 4], "59077356e": 0, "03": 0, "73881105e": 0, "11": 0, "79303854e": 0, "06": 0, "34813784e": 0, "04002859e": 0, "04": 0, "60750479e": 0, "05": 0, "67605334e": 0, "25171196e": 0, "73881104e": 0, "22082662e": 0, "14": 0, "64237500e": 0, "09": [0, 3, 4], "54833541e": 0, "37966733e": 0, "13": [0, 3], "52900681e": 0, "08": 0, "16215869e": 0, "66331834e": 0, "79303858e": 0, "24595981e": 0, "03938709e": 0, "47043395e": 0, "07": 0, "02364387e": 0, "81105534e": 0, "29976949e": 0, "34813786e": 0, "82376988e": 0, "53048679e": 0, "69427999e": 0, "96655655e": 0, "05333283e": 0, "37967094e": 0, "47043323e": 0, "53048569e": 0, "17508525e": 0, "31774245e": 0, "68420959e": 0, "78390944e": 0, "60750482e": 0, "31774311e": 0, "81657883e": 0, "02": 0, "78010037e": 0, "23202090e": 0, "16215870e": 0, "68420965e": 0, "55073927e": 0, "16053556e": 0, "78390947e": 0, "04009063e": 0, "standard": [0, 4], "deviat": [0, 4], "margin": [0, 4], "diagon": [0, 4], "01799284e": 0, "24": 0, "20982046e": 0, "35814698e": 0, "19": 0, "79710026e": 0, "08660507e": 0, "03479985e": 0, "57517708e": 0, "25836812e": 0, "ellips": 0, "intial": 0, "deriv": [0, 3, 4, 5], "ind1": 0, "ind2": 0, "inds_get": 0, "sub_mat": 0, "tupl": [0, 3, 4, 5], "reshap": [0, 4], "draw": 0, "b": 0, "flatten": [0, 4], "lam1": 0, "sqrt": 0, "lam2": 0, "theta": [0, 4], "elif": 0, "pi": [0, 3], "els": 0, "arctan2": 0, "t_val": 0, "linspac": 0, "1000": 0, "x": [0, 2, 4, 5], "co": 0, "sin": [0, 3], "y": [0, 5], "x_in": 0, "y_in": 0, "lt": 0, "line": [0, 3], "line2d": 0, "0x7fc8d2892310": 0, "gt": 0, "provid": [0, 3, 4, 5], "below": [0, 1, 2, 3, 4], "some": 0, "mai": [0, 1, 2], "given": [0, 3, 4, 5], "f_0": [0, 5], "_0": [0, 5], "ddot": [0, 5], "approxim": 0, "evolut": [0, 3], "quadrat": 0, "order": [0, 3, 4], "num": [0, 3], "15": 0, "random": 0, "uniform": 0, "001": 0, "002": 0, "17": 0, "28": 0, "arang": 0, "200": 0, "cast": [0, 5], "get_fgw": [0, 2, 5], "r": 0, "": [0, 3, 4, 5], "f_": 0, "gw": [0, 5], "text": 0, "39": 0, "12": 0, "m1": [0, 5], "m2": [0, 5], "d": [0, 4, 5], "kpc": [0, 5], "get_amplitud": [0, 2, 5], "37190427e": 0, "get_fdot": [0, 2, 5], "73056526e": 0, "get_n": [0, 2, 4, 5], "256": 0, "openmp": [0, 4, 5], "default": [0, 3, 4, 5], "thread": [0, 4, 5], "To": [0, 1, 2, 3, 4], "omp_set_num_thread": [0, 2, 5], "can": [0, 3, 4, 5], "access": [0, 4], "specif": [0, 4], "properti": [0, 3, 4], "softwar": [0, 1, 2], "michael_l_katz_2022_6500434": 0, "author": 0, "michael": [0, 1, 2], "l": [0, 3], "katz": [0, 1, 2], "titl": 0, "offici": 0, "public": 0, "releas": 0, "month": 0, "apr": 0, "2022": 0, "publish": 0, "v1": 0, "doi": 0, "5281": 0, "6500434": 0, "url": 0, "org": 0, "articl": 0, "2007if": 0, "34": 0, "j": 0, "test": [0, 4], "bayesian": 0, "model": [0, 3], "select": 0, "techniqu": 0, "astronomi": 0, "eprint": 0, "archiveprefix": 0, "primaryclass": 0, "gr": 0, "qc": 0, "1103": 0, "physrevd": 0, "76": 0, "083006": 0, "journal": 0, "phy": 0, "rev": 0, "volum": 0, "2007": [0, 3], "2018svj": 0, "tamanini": 0, "nicola": 0, "toonen": 0, "silvia": 0, "detect": 0, "hierarch": 0, "stellar": 0, "98": 0, "064012": 0, "2018": 0, "gravit": [1, 2, 3, 4], "c": [1, 2], "addit": [1, 2, 3, 4], "function": [1, 2, 4], "2205": [1, 2], "03461": [1, 2], "quick": [1, 2], "set": [1, 2, 4, 5], "instruct": [1, 2], "anaconda": [1, 2], "do": [1, 2, 3], "have": [1, 2, 3, 4], "creat": [1, 2, 4], "virtual": [1, 2], "environ": [1, 2], "note": [1, 2], "avail": [1, 2, 4], "conda": [1, 2], "compil": [1, 2], "window": [1, 2], "want": [1, 2], "probabl": [1, 2], "librari": [1, 2], "path": [1, 2], "py": [1, 2], "file": [1, 2], "gbgpu_env": [1, 2], "forg": [1, 2], "gcc_linux": [1, 2], "64": [1, 2], "gxx_linux": [1, 2], "gsl": [1, 2], "cython": [1, 2, 4], "scipi": [1, 2], "jupyt": [1, 2], "ipython": [1, 2], "h5py": [1, 2], "activ": [1, 2], "macosx": [1, 2], "substitut": [1, 2], "gxx_linu": [1, 2], "clang_osx": [1, 2], "clangxx_osx": [1, 2], "clone": [1, 2], "repositori": [1, 2], "git": [1, 2], "com": [1, 2], "cd": [1, 2], "make": [1, 2], "sure": [1, 2], "your": [1, 2], "usag": [1, 2, 3], "we": [1, 2, 3, 4], "gener": [1, 2, 4], "recommend": [1, 2, 3, 4], "everyth": [1, 2], "gcc": [1, 2], "g": [1, 2], "shown": [1, 2, 3], "exampl": [1, 2, 4], "here": [1, 2, 3], "help": [1, 2], "avoid": [1, 2], "link": [1, 2], "issu": [1, 2, 4], "own": [1, 2], "chosen": [1, 2], "inform": [1, 2, 4, 5], "capabl": [1, 2], "toolkit": [1, 2], "cupi": [1, 2, 4], "must": [1, 2, 3, 4, 5], "Be": [1, 2], "properli": [1, 2, 4, 5], "within": [1, 2], "correct": [1, 2], "nvcc": [1, 2], "cudahom": [1, 2], "variabl": [1, 2, 4, 5], "pip": [1, 2], "cuda92": [1, 2], "chang": [1, 2], "directori": [1, 2], "termin": [1, 2], "m": [1, 2, 3], "unittest": [1, 2], "discov": [1, 2], "semver": [1, 2], "tag": [1, 2], "project": [1, 2], "under": [1, 2], "gnu": [1, 2], "md": [1, 2], "gb": 2, "xp": [2, 3, 4, 5], "get_basis_tensor": [2, 4], "genwav": [2, 4], "genwavethird": [2, 4], "unpack_data_1": [2, 4], "xyz": [2, 4, 5], "get_ll_func": [2, 4], "n_max": [2, 4], "start_ind": [2, 4], "df": [2, 4], "d_d": [2, 4], "citat": [2, 3, 4], "e": [2, 3, 4, 5], "get_ll": [2, 4], "fill_global_templ": [2, 4], "generate_global_templ": [2, 4], "inject_sign": [2, 4], "extend": 2, "third": [2, 4], "bodi": [2, 4], "inclus": 2, "util": [2, 4], "third_body_factor": [2, 3], "get_t2": [2, 3], "wave": [2, 3, 4], "aet": [2, 5], "get_chirp_mass": [2, 5], "get_eta": [2, 5], "get_chirp_mass_from_f_fdot": [2, 5], "omp_get_num_thread": [2, 5], "cuda_set_devic": [2, 5], "ad": [2, 3], "astrophys": [2, 4], "orbit": [2, 3, 4], "around": [2, 3], "inner": [2, 3], "calcul": [2, 3, 4], "matrix": [2, 4], "instantan": [2, 3, 5], "frequenc": [2, 3, 4, 5], "amplitud": [2, 3, 4, 5], "slowli": 2, "evolv": 2, "sourc": [2, 4, 5], "dot": [2, 5], "f": [2, 5], "determin": [2, 3, 4, 5], "necessari": [2, 3], "sampl": [2, 3, 5], "rate": [2, 3, 5], "time": [2, 3, 4, 5], "domin": 2, "adjust": [2, 3, 4], "omp": [2, 5], "build": [3, 4], "thei": [3, 4], "describ": 3, "abstract": 3, "after": [3, 4], "alreadi": [3, 4], "abc": 3, "expand": 3, "classmethod": 3, "arg": [3, 4], "extra": 3, "transform": [3, 4], "them": [3, 4], "rest": 3, "where": 3, "copi": 3, "dealt": 3, "return": [3, 4, 5], "In": [3, 4, 5], "add_arg": 3, "proper": [3, 4], "doubl": [3, 4, 5], "1d": [3, 4, 5], "ndarrai": [3, 4, 5], "second": [3, 4, 5], "int": [3, 4, 5], "option": [3, 4, 5], "factor": [3, 4, 5], "compar": [3, 4], "valu": [3, 4, 5], "final": [3, 4], "onli": [3, 4], "each": [3, 4, 5], "binari": [3, 4, 5], "int32": [3, 4], "fi": 3, "xi": 3, "3d": [3, 4], "befor": 3, "appli": [3, 4], "spacecraft": 3, "updat": 3, "formal": 3, "goe": 3, "kdotp": 3, "directli": 3, "_construct_slow_part": 3, "larger": 3, "evalu": [3, 5], "respect": [3, 4], "long": 3, "built": 3, "recent": 3, "adapt": 3, "both": 3, "paper": 3, "galact": [3, 4], "compon": 3, "bool": [3, 4], "true": [3, 4], "attribut": 3, "get": [3, 4, 5], "site": 3, "veloc": 3, "defin": 3, "descript": [3, 4], "abov": [3, 4], "angular": 3, "pericent": 3, "integr": 3, "n2": 3, "n_2": 3, "frac": [3, 4, 5], "p_2": 3, "Not": 3, "rais": [3, 4, 5], "assertionerror": [3, 5], "wrong": 3, "per": 3, "get_u": 3, "invert": 3, "kepler": 3, "equat": 3, "anomali": 3, "u": 3, "mikkola": 3, "1987": 3, "referenc": 3, "tessmer": 3, "gopakumar": 3, "mean": 3, "radian": [3, 4], "get_phi": 3, "phi": 3, "sight": 3, "bar": 3, "periastron": 3, "get_vlo": 3, "lo": 3, "parab_step_et": 3, "t0": 3, "t0_old": 3, "differ": [3, 4], "caus": 3, "step": [3, 4], "done": 3, "parallel": [3, 5], "becaus": [3, 4], "just": [3, 4], "rather": 3, "than": 3, "od": 3, "serial": 3, "check": 3, "end": 3, "start": [3, 4], "due": 3, "get_alo": 3, "central": [3, 4], "get_f_deriv": 3, "50000": 3, "over": [3, 4], "cours": 3, "anyth": 3, "repres": [3, 4], "caluc": 3, "index": [3, 4], "differenc": 3, "word": 3, "dimens": 3, "should": [3, 4], "t_deriv": 3, "fill": [3, 4], "kwarg": [3, 4], "size": [3, 4], "mc": [3, 5], "omega2": 3, "phi2": 3, "beta": [3, 4], "third_mass_unit": 3, "mjup": 3, "third_period_unit": 3, "yr": 3, "go": 3, "total": 3, "mass": [3, 5], "solar": [3, 5], "unit": 3, "one": [3, 4], "three": 3, "euler": 3, "rotat": 3, "frame": 3, "two": [3, 4], "figur": 3, "str": 3, "jupit": 3, "msun": 3, "sec": 3, "associ": [3, 4], "gbgpu": 4, "object": 4, "form": 4, "inject": 4, "These": 4, "leverag": 4, "omp_num_thread": [4, 5], "environment": 4, "set_omp_num_thread": 4, "circular": 4, "desir": 4, "argmument": 4, "obj": 4, "maximum": 4, "harmon": 4, "mode": 4, "consid": 4, "indic": 4, "stream": 4, "q": 4, "list": 4, "fourier": 4, "bin": 4, "space": 4, "x_out": 4, "a_out": 4, "e_out": 4, "2d": 4, "complex": 4, "last": 4, "term": 4, "125829120": 4, "tdi2": 4, "call": 4, "ldc": 4, "longitutud": 4, "convert": 4, "spheric": 4, "flexibl": 4, "len": 4, "cadenc": 4, "produc": 4, "2nd": 4, "1st": 4, "technic": 4, "valid": 4, "dealth": 4, "those": 4, "equal": [4, 5], "arm": 4, "condit": 4, "valueerror": [4, 5], "psd": 4, "phase_margin": 4, "start_freq_ind": 4, "data_index": 4, "noise_index": 4, "log": 4, "complex128": 4, "analyz": 4, "data_length": 4, "power": 4, "spectral": 4, "densiti": 4, "nois": 4, "entri": 4, "subset": 4, "present": 4, "mutlipl": 4, "singl": 4, "being": 4, "100": 4, "zero": 4, "dict": 4, "pass": 4, "keyword": 4, "typeerror": 4, "while": 4, "group_index": 4, "group": 4, "buffer": 4, "tempalt": 4, "doe": 4, "hand": 4, "belong": 4, "output": 4, "place": 4, "arrang": 4, "resembl": 4, "transpos": 4, "how": 4, "self": 4, "correctli": 4, "wrap": 4, "fmax": 4, "span": 4, "work": 4, "parameter_transform": 4, "psd_func": 4, "psd_kwarg": 4, "return_gpu": 4, "math": 4, "dh": 4, "dlambda_i": 4, "h": 4, "lambda_i": 4, "2epsilon": 4, "epsilon": 4, "12epsilson": 4, "otherwis": 4, "map": 4, "squar": 4, "root": 4, "multipl": 4, "c_": 4, "ij": 4, "vec": 4, "_j": 4, "rel": 4, "dictionari": 4, "contain": 4, "kei": 4, "item": 4, "actual": 4, "same": [4, 5], "higher": 4, "numer": 4, "attemp": 4, "sensit": 4, "analysi": 4, "tool": 4, "dure": 4, "matric": 4, "modulenotfounderror": 4, "occur": 4, "NOT": 4, "z": 5, "hold": 5, "approx": 5, "assum": 5, "num_bin_al": 5, "ndim": 5, "case": 5, "chirp": 5, "symmetr": 5, "ratio": 5, "symetr": 5, "luminos": 5, "distanc": 5, "either": 5, "incorrect": 5, "waveform": 5, "enter": 5, "num_thread": 5, "dev": 5, "devic": 5}, "objects": {"gbgpu.gbgpu": [[4, 0, 1, "", "GBGPU"], [3, 0, 1, "", "InheritGBGPU"]], "gbgpu.gbgpu.GBGPU": [[4, 1, 1, "", "A"], [4, 1, 1, "", "E"], [4, 2, 1, "", "GenWave"], [4, 2, 1, "", "GenWaveThird"], [4, 2, 1, "", "N"], [4, 2, 1, "", "N_max"], [4, 1, 1, "", "X"], [4, 2, 1, "", "XYZ"], [4, 1, 1, "", "citation"], [4, 2, 1, "", "d_d"], [4, 2, 1, "", "df"], [4, 3, 1, "", "fill_global_template"], [4, 1, 1, "", "freqs"], [4, 3, 1, "", "generate_global_template"], [4, 2, 1, "", "get_basis_tensors"], [4, 3, 1, "", "get_ll"], [4, 2, 1, "", "get_ll_func"], [4, 3, 1, "", "information_matrix"], [4, 3, 1, "", "inject_signal"], [4, 2, 1, "", "num_bin"], [4, 3, 1, "", "run_wave"], [4, 2, 1, "", "start_inds"], [4, 2, 1, "", "unpack_data_1"], [4, 2, 1, "", "use_gpu"], [4, 2, 1, "", "xp"]], "gbgpu.gbgpu.InheritGBGPU": [[3, 3, 1, "", "add_to_argS"], [3, 3, 1, "", "prepare_additional_args"], [3, 3, 1, "", "shift_frequency"], [3, 3, 1, "", "special_get_N"]], "gbgpu.thirdbody": [[3, 0, 1, "", "GBGPUThirdBody"], [3, 4, 1, "", "get_T2"], [3, 4, 1, "", "third_body_factors"]], "gbgpu.thirdbody.GBGPUThirdBody": [[3, 3, 1, "", "add_to_argS"], [3, 1, 1, "", "citation"], [3, 3, 1, "", "get_aLOS"], [3, 3, 1, "", "get_f_derivatives"], [3, 3, 1, "", "get_phi"], [3, 3, 1, "", "get_u"], [3, 3, 1, "", "get_vLOS"], [3, 3, 1, "", "parab_step_ET"], [3, 3, 1, "", "prepare_additional_args"], [3, 3, 1, "", "shift_frequency"], [3, 3, 1, "", "special_get_N"]], "gbgpu.utils.utility": [[5, 4, 1, "", "AET"], [5, 4, 1, "", "cuda_set_device"], [5, 4, 1, "", "get_N"], [5, 4, 1, "", "get_amplitude"], [5, 4, 1, "", "get_chirp_mass"], [5, 4, 1, "", "get_chirp_mass_from_f_fdot"], [5, 4, 1, "", "get_eta"], [5, 4, 1, "", "get_fGW"], [5, 4, 1, "", "get_fdot"], [5, 4, 1, "", "omp_get_num_threads"], [5, 4, 1, "", "omp_set_num_threads"]]}, "objtypes": {"0": "py:class", "1": "py:property", "2": "py:attribute", "3": "py:method", "4": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "property", "Python property"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "method", "Python method"], "4": ["py", "function", "Python function"]}, "titleterms": {"gbgpu": [0, 1, 2, 3, 5], "tutori": [0, 2], "gener": 0, "galact": [0, 1, 2], "binari": [0, 1, 2], "waveform": [0, 1, 2, 3, 4], "ad": 0, "addit": 0, "gb": [0, 4], "astrophys": 0, "exampl": 0, "third": [0, 3], "bodi": [0, 3], "orbit": 0, "around": 0, "inner": 0, "calcul": 0, "inform": 0, "matrix": 0, "util": [0, 3, 5], "function": [0, 3, 5], "get": [0, 1, 2], "instantan": 0, "gravit": [0, 5], "wave": [0, 5], "frequenc": 0, "amplitud": 0, "slowli": 0, "evolv": 0, "sourc": 0, "dot": 0, "f": 0, "determin": 0, "necessari": 0, "sampl": 0, "rate": 0, "time": 0, "domin": 0, "adjust": 0, "omp": 0, "usag": 0, "c": 0, "citat": 0, "gpu": [1, 2], "cpu": [1, 2], "start": [1, 2], "prerequisit": [1, 2], "instal": [1, 2], "run": [1, 2], "test": [1, 2], "version": [1, 2], "author": [1, 2], "licens": [1, 2], "document": 2, "extend": 3, "inheritgbgpu": 3, "base": 3, "class": 3, "inclus": 3, "fast": 4, "likelihood": 4, "other": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "nbsphinx": 4, "sphinx": 57}, "alltitles": {"GBGPU Tutorial": [[0, "GBGPU-Tutorial"]], "Generating Galactic binary waveforms": [[0, "Generating-Galactic-binary-waveforms"]], "Adding additional GB astrophysics": [[0, "Adding-additional-GB-astrophysics"]], "Example: Third-body in orbit around the inner binary": [[0, "Example:-Third-body-in-orbit-around-the-inner-binary"]], "Calculating the Information Matrix": [[0, "Calculating-the-Information-Matrix"]], "Utility functions": [[0, "Utility-functions"]], "Get the instantaneous gravitational wave frequency": [[0, "Get-the-instantaneous-gravitational-wave-frequency"]], "Get amplitude (for slowly evolving source)": [[0, "Get-amplitude-(for-slowly-evolving-source)"]], "Get \\dot{f}": [[0, "Get-\\dot{f}"]], "Determine necessary sampling rate in the time-domin": [[0, "Determine-necessary-sampling-rate-in-the-time-domin"]], "Adjust OMP usage in C functions": [[0, "Adjust-OMP-usage-in-C-functions"]], "Citations": [[0, "Citations"]], "gbgpu: GPU/CPU Galactic Binary Waveforms": [[1, "gbgpu-gpu-cpu-galactic-binary-waveforms"], [2, "gbgpu-gpu-cpu-galactic-binary-waveforms"]], "Getting Started": [[1, "getting-started"], [2, "getting-started"]], "Prerequisites": [[1, "prerequisites"], [2, "prerequisites"]], "Installing": [[1, "installing"], [2, "installing"]], "Running the Tests": [[1, "running-the-tests"], [2, "running-the-tests"]], "Versioning": [[1, "versioning"], [2, "versioning"]], "Authors": [[1, "authors"], [2, "authors"]], "License": [[1, "license"], [2, "license"]], "Documentation:": [[2, null]], "Tutorial:": [[2, null]], "Extending GBGPU Waveforms": [[3, "extending-gbgpu-waveforms"]], "InheritGBGPU base class": [[3, "inheritgbgpu-base-class"]], "Third-body inclusion": [[3, "third-body-inclusion"]], "Third-body waveform": [[3, "third-body-waveform"]], "Third-body utility functions": [[3, "third-body-utility-functions"]], "Fast GB Waveforms and Likelihoods": [[4, "fast-gb-waveforms-and-likelihoods"]], "GBGPU Utility Functions": [[5, "gbgpu-utility-functions"]], "Gravitational-wave utilities": [[5, "gravitational-wave-utilities"]], "Other utilities": [[5, "other-utilities"]]}, "indexentries": {"gbgputhirdbody (class in gbgpu.thirdbody)": [[3, "gbgpu.thirdbody.GBGPUThirdBody"]], "inheritgbgpu (class in gbgpu.gbgpu)": [[3, "gbgpu.gbgpu.InheritGBGPU"]], "add_to_args() (gbgpu.gbgpu.inheritgbgpu method)": [[3, "gbgpu.gbgpu.InheritGBGPU.add_to_argS"]], "add_to_args() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.add_to_argS"]], "citation (gbgpu.thirdbody.gbgputhirdbody property)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.citation"]], "get_t2() (in module gbgpu.thirdbody)": [[3, "gbgpu.thirdbody.get_T2"]], "get_alos() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.get_aLOS"]], "get_f_derivatives() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.get_f_derivatives"]], "get_phi() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.get_phi"]], "get_u() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.get_u"]], "get_vlos() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.get_vLOS"]], "parab_step_et() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.parab_step_ET"]], "prepare_additional_args() (gbgpu.gbgpu.inheritgbgpu class method)": [[3, "gbgpu.gbgpu.InheritGBGPU.prepare_additional_args"]], "prepare_additional_args() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.prepare_additional_args"]], "shift_frequency() (gbgpu.gbgpu.inheritgbgpu method)": [[3, "gbgpu.gbgpu.InheritGBGPU.shift_frequency"]], "shift_frequency() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.shift_frequency"]], "special_get_n() (gbgpu.gbgpu.inheritgbgpu class method)": [[3, "gbgpu.gbgpu.InheritGBGPU.special_get_N"]], "special_get_n() (gbgpu.thirdbody.gbgputhirdbody method)": [[3, "gbgpu.thirdbody.GBGPUThirdBody.special_get_N"]], "third_body_factors() (in module gbgpu.thirdbody)": [[3, "gbgpu.thirdbody.third_body_factors"]], "a (gbgpu.gbgpu.gbgpu property)": [[4, "gbgpu.gbgpu.GBGPU.A"]], "e (gbgpu.gbgpu.gbgpu property)": [[4, "gbgpu.gbgpu.GBGPU.E"]], "gbgpu (class in gbgpu.gbgpu)": [[4, "gbgpu.gbgpu.GBGPU"]], "genwave (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.GenWave"]], "genwavethird (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.GenWaveThird"]], "n (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.N"]], "n_max (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.N_max"]], "x (gbgpu.gbgpu.gbgpu property)": [[4, "gbgpu.gbgpu.GBGPU.X"]], "xyz (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.XYZ"]], "citation (gbgpu.gbgpu.gbgpu property)": [[4, "gbgpu.gbgpu.GBGPU.citation"]], "d_d (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.d_d"]], "df (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.df"]], "fill_global_template() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.fill_global_template"]], "freqs (gbgpu.gbgpu.gbgpu property)": [[4, "gbgpu.gbgpu.GBGPU.freqs"]], "generate_global_template() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.generate_global_template"]], "get_basis_tensors (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.get_basis_tensors"]], "get_ll() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.get_ll"]], "get_ll_func (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.get_ll_func"]], "information_matrix() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.information_matrix"]], "inject_signal() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.inject_signal"]], "num_bin (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.num_bin"]], "run_wave() (gbgpu.gbgpu.gbgpu method)": [[4, "gbgpu.gbgpu.GBGPU.run_wave"]], "start_inds (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.start_inds"]], "unpack_data_1 (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.unpack_data_1"]], "use_gpu (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.use_gpu"]], "xp (gbgpu.gbgpu.gbgpu attribute)": [[4, "gbgpu.gbgpu.GBGPU.xp"]], "aet() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.AET"]], "cuda_set_device() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.cuda_set_device"]], "get_n() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_N"]], "get_amplitude() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_amplitude"]], "get_chirp_mass() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_chirp_mass"]], "get_chirp_mass_from_f_fdot() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_chirp_mass_from_f_fdot"]], "get_eta() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_eta"]], "get_fgw() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_fGW"]], "get_fdot() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.get_fdot"]], "omp_get_num_threads() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.omp_get_num_threads"]], "omp_set_num_threads() (in module gbgpu.utils.utility)": [[5, "gbgpu.utils.utility.omp_set_num_threads"]]}})