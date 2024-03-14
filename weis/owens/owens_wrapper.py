# juliaup update
# pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install juliacall
from juliacall import Main as jl
from juliacall import Pkg as jlPkg

OWENS_directory = '/Users/dzalkind/Tools/Cross_Flow/OWENS.jl'

jlPkg.activate(OWENS_directory)  # relative path to the folder where `MyPack/Project.toml` should be used here 

jl.seval("using OWENS")

testdata = jl.OWENS.runOWENS(jl.OWENS.MasterInput("./sampleOWENS.yml"),"./").to_numpy()