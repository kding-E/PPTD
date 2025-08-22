contract Token {
    Transfer _transfer;
    function tokenTransfer() public returns (bytes memory) {
        bytes memory method = abi.encodeWithSignature("Transfer()");
        (bool success, bytes memory returnData) = address(_transfer).call(method);
        return returnData; }
    function setTransfer(address payable ad) public {
        _transfer=Transfer(ad); }
}
contract Transfer {
    Token _token;
    fallback() external payable{
        bytes memory method = abi.encodeWithSignature("tokenTransfer()");
        (bool success, bytes memory returnData) = address(_token).call(method);
    }
    function transfer() public returns (bytes memory) {
        (bool success, bytes memory returnData) = payable(msg.sender).call{value: 1}("");
        return returnData; }
    function setTransfer(address ad) public {
        _token=Token(ad); }
}