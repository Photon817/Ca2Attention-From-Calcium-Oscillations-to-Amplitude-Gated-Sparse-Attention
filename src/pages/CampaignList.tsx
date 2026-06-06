import { Link } from 'react-router-dom';
import { Plus, Search, Filter } from 'lucide-react';
import { campaigns } from '../mockData';
import { CampaignBadge } from '../components/StatusBadge';
import Nav from '../components/Nav';

export default function CampaignList() {
  return (
    <div>
      <Nav items={[{ label: 'Campaigns' }]} />
      <div className="flex items-center justify-between mb-5">
        <h2 className="text-xl font-bold text-slate-900">Campaign List</h2>
        <button className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
          <Plus size={16} /> 新建 Campaign
        </button>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1 max-w-md">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input placeholder="搜索 Campaign 或品牌" className="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500" />
        </div>
        <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">
          <Filter size={14} /> 筛选
        </button>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-slate-500">
            <tr>
              <th className="text-left px-4 py-3 font-medium">Campaign</th>
              <th className="text-left px-4 py-3 font-medium">品牌</th>
              <th className="text-left px-4 py-3 font-medium">状态</th>
              <th className="text-left px-4 py-3 font-medium">预算</th>
              <th className="text-left px-4 py-3 font-medium">已用</th>
              <th className="text-left px-4 py-3 font-medium">周期</th>
              <th className="text-left px-4 py-3 font-medium">负责人</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {campaigns.map((c) => (
              <tr key={c.id} className="hover:bg-slate-50">
                <td className="px-4 py-3 font-medium text-slate-900">
                  <Link to={`/campaigns/${c.id}`} className="hover:text-brand-600">{c.name}</Link>
                </td>
                <td className="px-4 py-3 text-slate-600">{c.brand}</td>
                <td className="px-4 py-3"><CampaignBadge status={c.status} /></td>
                <td className="px-4 py-3 text-slate-700">¥{c.budget.toLocaleString()}</td>
                <td className="px-4 py-3 text-slate-700">¥{c.spent.toLocaleString()}</td>
                <td className="px-4 py-3 text-slate-600 text-xs">{c.startDate} ~ {c.endDate}</td>
                <td className="px-4 py-3 text-slate-600">{c.manager}</td>
                <td className="px-4 py-3">
                  <Link to={`/campaigns/${c.id}`} className="text-xs text-brand-600 hover:underline">详情</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}