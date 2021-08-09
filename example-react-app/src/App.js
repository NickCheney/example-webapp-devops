import logo from './logo.svg';
import './App.css';
import Symbols from './Symbols';
import StockChart from './StockChart';

function App() {
  return (
    <div className="App">
      
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h1 className="site-title">S&P 500 Stonks</h1>
      </header>
      <nav className="page-menu">
        <ul className="page-list">
          <li><a href="#top" className="menu-item">Stock List</a></li>
          <li><a href="#top" className="menu-item">Trending up</a></li>
          <li><a href="#top" className="menu-item">Trending down</a></li>
        </ul>
      </nav>
      <div className="main-page">
        <br/>
        <h2 className="App-subheading">Stock List</h2>
        <Symbols />
      </div>
    </div>
  );
}

export default App;
