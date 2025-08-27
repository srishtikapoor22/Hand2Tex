# Go into the kaggle_math folder
Set-Location "data\kaggle_math"

# Rename special symbol folders to safer names
Rename-Item -LiteralPath "+"        -NewName "plus"
Rename-Item -LiteralPath "MINUS"    -NewName "minus"
Rename-Item -LiteralPath "div"      -NewName "divide"
Rename-Item -LiteralPath ","        -NewName "comma"
Rename-Item -LiteralPath "!"        -NewName "factorial"
Rename-Item -LiteralPath "{"        -NewName "lbrace"
Rename-Item -LiteralPath "}"        -NewName "rbrace"
Rename-Item -LiteralPath "["        -NewName "lbrack"
Rename-Item -LiteralPath "]"        -NewName "rbrack"
Rename-Item -LiteralPath "("        -NewName "lparen"
Rename-Item -LiteralPath ")"        -NewName "rparen"
Rename-Item -LiteralPath "forward_slash" -NewName "slash"
