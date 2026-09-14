import "./ChartCard.css";

export default function ChartCard({ title, children }) {
  return (
    <section className="chart-card">
      <h2 className="chart-card-title">{title}</h2>
      <div className="chart-placeholder">
        {/* Real chart component (ECharts via echarts-for-react) drops in here. */}
        {children ?? <span>Chart coming soon</span>}
      </div>
    </section>
  );
}
