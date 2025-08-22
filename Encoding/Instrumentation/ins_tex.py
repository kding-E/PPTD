import re
def instrument_solidity_code(solidity_code):
    # Match contract definitions
    contract_pattern = re.compile(r"(contract\s+\w+\s*{)(.*?)(?=contract|$)", re.DOTALL)
    
    # Check if there are call, delegatecall, or staticcall invocations in each contract definition
    def insert_tex_if_call(match):
        contract_start = match.group(1)
        contract_body = match.group(2)

        # print(contract_body)
   
        
        # Check if there are call, delegatecall, or staticcall invocations in contract_body
        if "call(" in contract_body or "delegatecall(" in contract_body or "staticcall(" in contract_body:
            # Insert tex state variable definition after contract_start
            return contract_start + "\n    uint256 private tex = 0;\n" + contract_body 
        else:
            # If there are no call invocations, no modification is needed
            return contract_start + contract_body #+ contract_end
    
    # Use regular expression's sub method for replacement
    instrumented_code = contract_pattern.sub(insert_tex_if_call, solidity_code)
    
    print(instrumented_code)
    
    # Match contract definitions
    contract_pattern = re.compile(r"([^;^{]*(\bcall\(|\bdelegatecall\(|\bstaticcall\()[^;]*);", re.DOTALL)
    
    # Insert tex state variable in each contract definition
    def insert_tex(match):
        return "\n    tex++;\n" + match.group(0) + f"\n    tex--; \n if(tex!=0){{revert();}}\n"
    
    # Use regular expression's sub method for replacement
    instrumented_code = contract_pattern.sub(insert_tex, instrumented_code)
    print(instrumented_code)
    return instrumented_code
