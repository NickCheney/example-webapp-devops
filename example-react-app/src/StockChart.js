/*const {PythonShell} = require('python-shell');*/
import { LineChart, Line, CartesianGrid, XAxis, YAxis, AreaChart, Area, Tooltip } from 'recharts';
const rangeData = [
    {
      "day": "05-01",
      "temperature": 10
    },
    {
      "day": "05-02",
      "temperature": 15
    },
    {
      "day": "05-03",
      "temperature": 12
    },
    {
      "day": "05-04",
      "temperature": 12
    },
    {
      "day": "05-05",
      "temperature": 16
    },
    {
      "day": "05-06",
      "temperature": 16
    },
    {
      "day": "05-07",
      "temperature": 12
    },
    {
      "day": "05-08",
      "temperature": 8
    },
    {
      "day": "05-09",
      "temperature": 5
    }
  ]

function StockChart() {
    const renderLineChart = (
        <AreaChart
  width={600}
  height={300}
  data={rangeData}
  margin={{
    top: 20, right: 20, bottom: 0, left: 0,
  }}
  className = "stock-chart"
>
  <XAxis dataKey="day" />
  <YAxis />
  <Area dataKey="temperature" stroke="#326d7e" fill="#61dafb" />
  <Tooltip />
  <CartesianGrid vertical = {false} />
</AreaChart>
      );
    return(renderLineChart)
}


export default StockChart;