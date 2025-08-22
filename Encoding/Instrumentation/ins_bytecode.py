def insert_bytecode_before_exits(original_bytecode, instrumentation_bytecode, fix_jumps=True, validate=True):
    """
    Insert specified bytecode before all exit opcodes (RETURN, STOP, REVERT) in smart contract bytecode
    
    Args:
        original_bytecode: Original contract bytecode (hex string, case insensitive)
        instrumentation_bytecode: Bytecode to be inserted (hex string, case insensitive)
        fix_jumps: Whether to fix jump offsets (default True)
        validate: Whether to perform input validation (default True)
    
    Returns:
        tuple: (Modified bytecode (hex string), insertion information dictionary)
        
    Raises:
        ValueError: Raised when input parameters are invalid
    """
    try:
        # Input validation
        if validate:
            _validate_bytecode_input(original_bytecode, instrumentation_bytecode)
        
        # Normalize input (convert to lowercase, remove spaces)
        original_bytecode = original_bytecode.lower().replace(' ', '').replace('0x', '')
        instrumentation_bytecode = instrumentation_bytecode.lower().replace(' ', '').replace('0x', '')
        
        bytecode = bytearray.fromhex(original_bytecode)
        instrumentation = bytearray.fromhex(instrumentation_bytecode)
        
        # Analyze original bytecode to get instruction mapping
        instruction_map = _analyze_bytecode_instructions(bytecode)
        
        # Target opcodes: RETURN (0xf3), REVERT (0xfd), STOP (0x00)
        target_opcodes = {0xf3: 'RETURN', 0xfd: 'REVERT', 0x00: 'STOP'}
        
        # Find positions of all exit opcodes (only insert at actual instructions, not data)
        exit_positions = []
        for pos, opcode in instruction_map.items():
            if opcode in target_opcodes:
                exit_positions.append({
                    'position': pos,
                    'opcode': opcode,
                    'name': target_opcodes[opcode]
                })
        
        if not exit_positions:
            print("Warning: No exit opcodes found (RETURN/REVERT/STOP)")
            return original_bytecode, {'inserted_count': 0, 'positions': []}
        
        # Sort by position in reverse order, insert from back to front
        exit_positions.sort(key=lambda x: x['position'], reverse=True)
        
        # Execute insertion
        modified_bytecode = bytearray(bytecode)
        insertion_info = []
        
        for exit_info in exit_positions:
            pos = exit_info['position']
            # Recalculate current position (considering previous insertions)
            adjusted_pos = _calculate_adjusted_position(pos, insertion_info, len(instrumentation))
            
            # Insert bytecode
            modified_bytecode = modified_bytecode[:adjusted_pos] + instrumentation + modified_bytecode[adjusted_pos:]
            
            insertion_info.append({
                'original_position': pos,
                'adjusted_position': adjusted_pos,
                'opcode_name': exit_info['name'],
                'inserted_length': len(instrumentation)
            })
        
        # Fix jump offsets
        if fix_jumps and insertion_info:
            try:
                modified_bytecode = _fix_jump_offsets_advanced(modified_bytecode, insertion_info, instruction_map)
            except Exception as e:
                print(f"Warning: Advanced jump offset fixing failed: {e}")
                # Try basic fix
                try:
                    modified_bytecode = _fix_jump_offsets_basic(modified_bytecode, insertion_info)
                except Exception as e2:
                    print(f"Warning: Basic jump fixing also failed: {e2}")
        
        result_info = {
            'inserted_count': len(insertion_info),
            'positions': insertion_info,
            'original_length': len(bytecode),
            'final_length': len(modified_bytecode),
            'instrumentation_length': len(instrumentation)
        }
        
        return modified_bytecode.hex(), result_info
        
    except Exception as e:
        raise ValueError(f"Bytecode insertion failed: {str(e)}")


def _validate_bytecode_input(original_bytecode, instrumentation_bytecode):
    """Validate the validity of input parameters"""
    if not original_bytecode or not isinstance(original_bytecode, str):
        raise ValueError("Original bytecode must be a non-empty string")
    
    if not instrumentation_bytecode or not isinstance(instrumentation_bytecode, str):
        raise ValueError("Instrumentation bytecode must be a non-empty string")
    
    # Clean and check hex format
    clean_original = original_bytecode.lower().replace(' ', '').replace('0x', '')
    clean_instrumentation = instrumentation_bytecode.lower().replace(' ', '').replace('0x', '')
    
    if len(clean_original) % 2 != 0:
        raise ValueError("Original bytecode length must be even")
    
    if len(clean_instrumentation) % 2 != 0:
        raise ValueError("Instrumentation bytecode length must be even")
    
    # Check if valid hexadecimal
    try:
        bytearray.fromhex(clean_original)
        bytearray.fromhex(clean_instrumentation)
    except ValueError:
        raise ValueError("Input must be valid hexadecimal strings")


def _analyze_bytecode_instructions(bytecode):
    """
    Analyze bytecode to distinguish between instructions and data
    
    Returns:
        dict: {position: opcode} mapping, containing only actual instruction positions
    """
    instruction_map = {}
    i = 0
    
    while i < len(bytecode):
        opcode = bytecode[i]
        instruction_map[i] = opcode
        
        # Handle PUSH instructions (0x60-0x7f), skip subsequent data bytes
        if 0x60 <= opcode <= 0x7f:
            push_size = opcode - 0x5f
            i += push_size + 1  # Skip PUSH instruction and its data
        else:
            i += 1
    
    return instruction_map


def _calculate_adjusted_position(original_pos, insertion_info, instrumentation_length):
    """Calculate adjusted position considering previous insertions"""
    adjustment = 0
    for info in insertion_info:
        if info['original_position'] > original_pos:
            adjustment += info['inserted_length']
    return original_pos + adjustment


def _fix_jump_offsets_advanced(bytecode, insertion_info, instruction_map):
    """
    Advanced jump offset fixing function
    
    Args:
        bytecode: Modified bytecode
        insertion_info: List of insertion information
        instruction_map: Original instruction mapping
    
    Returns:
        Bytecode with fixed jump offsets
    """
    modified_bytecode = bytearray(bytecode)
    
    # Create address mapping table: original address -> new address
    address_mapping = _create_address_mapping(insertion_info, len(bytecode))
    
    i = 0
    while i < len(modified_bytecode):
        opcode = modified_bytecode[i]
        
        # Handle PUSH instructions, these instructions may contain jump addresses
        if 0x60 <= opcode <= 0x7f:  # PUSH1 to PUSH32
            push_size = opcode - 0x5f
            if i + push_size < len(modified_bytecode):
                try:
                    # Extract pushed data
                    push_data = int.from_bytes(modified_bytecode[i+1:i+1+push_size], 'big')
                    
                    # Check if it's a valid jump address (within reasonable range)
                    if 0 <= push_data < len(bytecode) * 2:  # Heuristic: address should be within reasonable range
                        # Check if this address needs adjustment
                        new_address = _calculate_new_address(push_data, insertion_info)
                        
                        if new_address != push_data:
                            # Check if new address is within push instruction's capacity
                            max_value = (256 ** push_size) - 1
                            if new_address <= max_value:
                                # Update address
                                modified_bytecode[i+1:i+1+push_size] = new_address.to_bytes(push_size, 'big')
                except (ValueError, OverflowError) as e:
                    # Conversion failed, skip
                    pass
                
                i += push_size + 1
            else:
                i += 1
        else:
            i += 1
    
    return modified_bytecode


def _fix_jump_offsets_basic(bytecode, insertion_info):
    """
    Basic jump offset fixing function (safe version)
    
    Args:
        bytecode: Modified bytecode
        insertion_info: List of insertion information
    
    Returns:
        Bytecode with fixed jump offsets
    """
    modified_bytecode = bytearray(bytecode)
    
    # Only handle PUSH1 instructions (0x60), as they are most commonly used for jump addresses
    i = 0
    while i < len(modified_bytecode):
        if modified_bytecode[i] == 0x60:  # PUSH1
            if i + 1 < len(modified_bytecode):
                try:
                    address = modified_bytecode[i + 1]
                    new_address = _calculate_new_address(address, insertion_info)
                    
                    # Ensure new address is within PUSH1 range (0-255)
                    if 0 <= new_address <= 255:
                        modified_bytecode[i + 1] = new_address
                except:
                    pass  # Safely ignore errors
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return modified_bytecode


def _create_address_mapping(insertion_info, original_length):
    """Create mapping from original addresses to new addresses"""
    mapping = {}
    
    # Sort by original position
    sorted_insertions = sorted(insertion_info, key=lambda x: x['original_position'])
    
    for addr in range(original_length):
        new_addr = addr
        for info in sorted_insertions:
            if info['original_position'] <= addr:
                new_addr += info['inserted_length']
        mapping[addr] = new_addr
    
    return mapping


def _calculate_new_address(original_address, insertion_info):
    """
    Calculate new address corresponding to original address
    
    Args:
        original_address: Original address
        insertion_info: List of insertion information
    
    Returns:
        Adjusted new address
    """
    adjustment = 0
    for info in insertion_info:
        # If insertion position is before or equal to target address, need adjustment
        if info['original_position'] <= original_address:
            adjustment += info['inserted_length']
    
    return original_address + adjustment


def _is_jump_instruction(opcode):
    """
    Check if opcode is a jump-related instruction
    
    Args:
        opcode: Opcode
    
    Returns:
        bool: Whether it is a jump instruction
    """
    # PUSH1-PUSH32 (0x60-0x7f) used for pushing jump addresses
    # JUMP (0x56), JUMPI (0x57) are jump instructions
    # JUMPDEST (0x5b) is jump destination
    return (0x60 <= opcode <= 0x7f) or opcode in [0x56, 0x57, 0x5b]








def create_custom_instrumentation(instruction_list):
    """
    Create custom bytecode based on instruction list
    
    Args:
        instruction_list: List of instructions, each element is (opcode, data) tuple
    
    Returns:
        Generated bytecode (hex string)
    
    Examples:
        # Create a simple storage operation
        instructions = [
            ('PUSH1', '01'),      # Push value 1
            ('PUSH1', '00'),      # Push storage position 0
            ('SSTORE', None)      # Store
        ]
        bytecode = create_custom_instrumentation(instructions)
    """
    opcode_map = {
        'STOP': '00', 'ADD': '01', 'MUL': '02', 'SUB': '03', 'DIV': '04',
        'SDIV': '05', 'MOD': '06', 'SMOD': '07', 'ADDMOD': '08', 'MULMOD': '09',
        'EXP': '0a', 'SIGNEXTEND': '0b', 'LT': '10', 'GT': '11', 'SLT': '12',
        'SGT': '13', 'EQ': '14', 'ISZERO': '15', 'AND': '16', 'OR': '17',
        'XOR': '18', 'NOT': '19', 'BYTE': '1a', 'SHL': '1b', 'SHR': '1c',
        'SAR': '1d', 'SHA3': '20', 'ADDRESS': '30', 'BALANCE': '31',
        'ORIGIN': '32', 'CALLER': '33', 'CALLVALUE': '34', 'CALLDATALOAD': '35',
        'CALLDATASIZE': '36', 'CALLDATACOPY': '37', 'CODESIZE': '38',
        'CODECOPY': '39', 'GASPRICE': '3a', 'EXTCODESIZE': '3b',
        'EXTCODECOPY': '3c', 'RETURNDATASIZE': '3d', 'RETURNDATACOPY': '3e',
        'EXTCODEHASH': '3f', 'BLOCKHASH': '40', 'COINBASE': '41',
        'TIMESTAMP': '42', 'NUMBER': '43', 'DIFFICULTY': '44', 'GASLIMIT': '45',
        'CHAINID': '46', 'SELFBALANCE': '47', 'BASEFEE': '48', 'POP': '50',
        'MLOAD': '51', 'MSTORE': '52', 'MSTORE8': '53', 'SLOAD': '54',
        'SSTORE': '55', 'JUMP': '56', 'JUMPI': '57', 'PC': '58',
        'MSIZE': '59', 'GAS': '5a', 'JUMPDEST': '5b', 'PUSH1': '60',
        'PUSH2': '61', 'PUSH3': '62', 'PUSH4': '63', 'PUSH5': '64',
        'PUSH6': '65', 'PUSH7': '66', 'PUSH8': '67', 'PUSH9': '68',
        'PUSH10': '69', 'PUSH11': '6a', 'PUSH12': '6b', 'PUSH13': '6c',
        'PUSH14': '6d', 'PUSH15': '6e', 'PUSH16': '6f', 'PUSH17': '70',
        'PUSH18': '71', 'PUSH19': '72', 'PUSH20': '73', 'PUSH21': '74',
        'PUSH22': '75', 'PUSH23': '76', 'PUSH24': '77', 'PUSH25': '78',
        'PUSH26': '79', 'PUSH27': '7a', 'PUSH28': '7b', 'PUSH29': '7c',
        'PUSH30': '7d', 'PUSH31': '7e', 'PUSH32': '7f', 'DUP1': '80',
        'DUP2': '81', 'DUP3': '82', 'DUP4': '83', 'DUP5': '84',
        'DUP6': '85', 'DUP7': '86', 'DUP8': '87', 'DUP9': '88',
        'DUP10': '89', 'DUP11': '8a', 'DUP12': '8b', 'DUP13': '8c',
        'DUP14': '8d', 'DUP15': '8e', 'DUP16': '8f', 'SWAP1': '90',
        'SWAP2': '91', 'SWAP3': '92', 'SWAP4': '93', 'SWAP5': '94',
        'SWAP6': '95', 'SWAP7': '96', 'SWAP8': '97', 'SWAP9': '98',
        'SWAP10': '99', 'SWAP11': '9a', 'SWAP12': '9b', 'SWAP13': '9c',
        'SWAP14': '9d', 'SWAP15': '9e', 'SWAP16': '9f', 'LOG0': 'a0',
        'LOG1': 'a1', 'LOG2': 'a2', 'LOG3': 'a3', 'LOG4': 'a4',
        'CREATE': 'f0', 'CALL': 'f1', 'CALLCODE': 'f2', 'RETURN': 'f3',
        'DELEGATECALL': 'f4', 'CREATE2': 'f5', 'STATICCALL': 'fa',
        'REVERT': 'fd', 'SELFDESTRUCT': 'ff'
    }
    
    bytecode = ""
    for instruction, data in instruction_list:
        if instruction.upper() in opcode_map:
            bytecode += opcode_map[instruction.upper()]
            if data is not None:
                # Ensure data is even-length hex string
                clean_data = data.replace('0x', '').replace(' ', '')
                if len(clean_data) % 2 != 0:
                    clean_data = '0' + clean_data
                bytecode += clean_data
        else:
            raise ValueError(f"Unknown instruction: {instruction}")
    
    return bytecode


def analyze_bytecode_exits(bytecode_hex):
    """
    Analyze all exit points in bytecode
    
    Args:
        bytecode_hex: Bytecode (hex string)
    
    Returns:
        dict: Dictionary containing exit point analysis information
    """
    try:
        bytecode = bytearray.fromhex(bytecode_hex.replace(' ', '').replace('0x', ''))
        instruction_map = _analyze_bytecode_instructions(bytecode)
        
        exit_opcodes = {0xf3: 'RETURN', 0xfd: 'REVERT', 0x00: 'STOP'}
        exits = []
        
        for pos, opcode in instruction_map.items():
            if opcode in exit_opcodes:
                exits.append({
                    'position': pos,
                    'opcode': hex(opcode),
                    'name': exit_opcodes[opcode],
                    'context': _get_instruction_context(bytecode, pos)
                })
        
        return {
            'total_exits': len(exits),
            'exits': exits,
            'bytecode_length': len(bytecode),
            'instruction_count': len(instruction_map)
        }
        
    except Exception as e:
        return {'error': str(e)}


def _get_instruction_context(bytecode, position, context_size=5):
    """Get instruction context"""
    start = max(0, position - context_size)
    end = min(len(bytecode), position + context_size + 1)
    context = bytecode[start:end]
    return {
        'bytes_before': context[:position-start].hex() if position > start else '',
        'target_byte': f"{bytecode[position]:02x}",
        'bytes_after': context[position-start+1:].hex() if position < end-1 else ''
    }

