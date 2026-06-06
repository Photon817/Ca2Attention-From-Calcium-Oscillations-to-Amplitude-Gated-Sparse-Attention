import { HashRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import CampaignList from './pages/CampaignList';
import CampaignDetail from './pages/CampaignDetail';
import PreviewReview from './pages/PreviewReview';
import PMSummary from './pages/PMSummary';

export default function App() {
  return (
    <HashRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/campaigns" element={<CampaignList />} />
          <Route path="/campaigns/:id" element={<CampaignDetail />} />
          <Route path="/campaigns/:id/preview" element={<PreviewReview />} />
          <Route path="/campaigns/:id/summary" element={<PMSummary />} />
        </Routes>
      </Layout>
    </HashRouter>
  );
}