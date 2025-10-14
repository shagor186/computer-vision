const StatCard = ({ title, value }) => {
  return (
    <div className="bg-white p-4 rounded shadow flex flex-col items-center">
      <h3 className="text-gray-500">{title}</h3>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
};

export default StatCard;
