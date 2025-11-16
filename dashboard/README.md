# 🎨 Futuristic Fraud Detection Dashboard

A modern, animated web dashboard for monitoring fraud detection with Temporal Graph Neural Networks (TGNN). Built with Next.js 14, Framer Motion, and 3D visualizations.

![Dashboard](https://img.shields.io/badge/Next.js-14-black?logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)
![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?logo=tailwindcss)
![Framer Motion](https://img.shields.io/badge/Framer%20Motion-11-ff69b4)

---

## ✨ Features

### 🎭 **Animated UI Components**
- **Smooth transitions** with Framer Motion
- **Hover effects** and micro-interactions
- **Real-time metric updates** with animated counters
- **Gradient text** and glow effects
- **Responsive grid layouts**

### 📊 **Data Visualization**
- **Interactive charts** (Recharts)
- **Model comparison graphs** (TGN vs MPTGNN vs Baselines)
- **ROC curves** and confusion matrices
- **Real-time transaction stream**
- **Plotly** integration for scientific plots

### 🌐 **3D Visualizations**
- **Three.js** graph network (React Three Fiber)
- **Interactive fraud network** exploration
- **Node clustering** visualization
- **Temporal edge animations**

### 🔌 **Embedding Support**
- **Weights & Biases** dashboard embedding (iframe)
- **Plotly** chart embedding
- **Custom analytics** widgets
- **External tool integration**

### 🚀 **Real-time Monitoring**
- **Live transaction feed**
- **Fraud detection alerts**
- **System status indicators**
- **WebSocket** support (coming soon)

---

## 🏗️ Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe code
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Animation library
- **shadcn/ui** - Beautiful, accessible components
- **Lucide Icons** - Modern icon set

### Visualization Libraries
- **Recharts** - Composable chart library
- **Plotly.js** - Scientific graphing library
- **React Three Fiber** - 3D rendering (Three.js for React)
- **@react-three/drei** - Three.js helpers

### Backend Integration
- **FastAPI** (Python) - ML model serving
- **SWR** - Data fetching and caching
- **Axios** - HTTP client

---

## 📦 Installation

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.12+ (for ML backend)

### 1. Install Dashboard Dependencies

```powershell
cd dashboard
npm install
```

This will install:
- Next.js, React, TypeScript
- Framer Motion, Tailwind CSS
- Recharts, Plotly, Three.js
- All UI component libraries

### 2. Install Additional Dependencies (if needed)

```powershell
# For tailwindcss-animate plugin
npm install tailwindcss-animate

# For React types (if missing)
npm install --save-dev @types/react @types/react-dom
```

---

## 🚀 Running the Dashboard

### Development Mode

```powershell
cd dashboard
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```powershell
npm run build
npm start
```

### Run Linting

```powershell
npm run lint
```

---

## 📂 Project Structure

```
dashboard/
├── app/                        # Next.js App Router
│   ├── layout.tsx             # Root layout (dark theme)
│   ├── page.tsx               # Dashboard homepage
│   ├── globals.css            # Global styles + animations
│   ├── monitoring/            # Live monitoring page
│   ├── graph/                 # 3D graph visualization
│   ├── analytics/             # Model analytics
│   └── experiments/           # W&B experiments embedding
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx        # Animated sidebar navigation
│   │   └── Header.tsx         # Top header with search
│   │
│   ├── dashboard/
│   │   ├── MetricsGrid.tsx    # Animated metric cards
│   │   ├── ModelComparison.tsx # Performance charts
│   │   └── RecentDetections.tsx # Transaction feed
│   │
│   ├── visualizations/
│   │   ├── GraphVisualization3D.tsx  # Three.js network
│   │   ├── ROCCurve.tsx              # Plotly ROC curve
│   │   └── ConfusionMatrix.tsx       # Heatmap
│   │
│   └── ui/                    # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       └── ...
│
├── lib/
│   ├── api.ts                 # API client for ML backend
│   └── utils.ts               # Utility functions
│
├── public/                    # Static assets
├── tailwind.config.js         # Tailwind configuration
├── tsconfig.json              # TypeScript config
└── package.json
```

---

## 🎨 Dashboard Pages

### 1. **Main Dashboard** (`/`)
- Overview metrics (transactions, fraud detected, accuracy)
- Model performance comparison chart
- Recent fraud detections feed
- System status indicators

### 2. **Live Monitoring** (`/monitoring`)
- Real-time transaction stream
- Live fraud detection alerts
- Animated transaction flow
- Geographic heatmap (coming soon)

### 3. **Graph Visualization** (`/graph`)
- **3D interactive fraud network** (Three.js)
- Node clustering by fraud risk
- Temporal edge evolution
- Click nodes for transaction details

### 4. **Model Analytics** (`/analytics`)
- ROC curves (TGN, MPTGNN, baselines)
- Confusion matrices
- Feature importance analysis
- Path attention weights (MPTGNN)

### 5. **Experiments** (`/experiments`)
- Embedded W&B dashboard
- Hyperparameter sweep results
- Training run comparison
- Export to reports

---

## 🔌 Embedding External Dashboards

### Weights & Biases

Create a new page to embed your W&B project:

```tsx
// app/experiments/page.tsx
export default function ExperimentsPage() {
  return (
    <div className="w-full h-screen">
      <iframe
        src="https://wandb.ai/YOUR-USERNAME/fraud-detection-phase1?workspace=user-YOUR-USERNAME"
        className="w-full h-full border-0"
        allow="fullscreen"
      />
    </div>
  );
}
```

### Plotly Charts

```tsx
import Plot from 'react-plotly.js';

export default function ROCCurve() {
  return (
    <Plot
      data={[
        {
          x: fpr,
          y: tpr,
          type: 'scatter',
          mode: 'lines',
          line: { color: '#3b82f6', width: 3 },
          name: 'TGN (AUC=0.958)',
        },
      ]}
      layout={{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#f3f4f6' },
      }}
    />
  );
}
```

---

## 🔗 Connecting to ML Backend

### Start FastAPI Backend

```powershell
# In project root
cd api
uvicorn main:app --reload
```

### Fetch Data from Dashboard

```typescript
// lib/api.ts
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const fetchMetrics = async () => {
  const response = await axios.get(`${API_URL}/api/metrics`);
  return response.data;
};

export const fetchPredictions = async (transactionId: string) => {
  const response = await axios.post(`${API_URL}/api/predict`, {
    transaction_id: transactionId,
  });
  return response.data;
};
```

### Use in Component with SWR

```tsx
import useSWR from 'swr';
import { fetchMetrics } from '@/lib/api';

export default function MetricsGrid() {
  const { data, error } = useSWR('/api/metrics', fetchMetrics, {
    refreshInterval: 3000, // Refresh every 3 seconds
  });

  if (error) return <div>Failed to load</div>;
  if (!data) return <div>Loading...</div>;

  return <div>{/* Render metrics */}</div>;
}
```

---

## 🎨 Customization

### Theme Colors

Edit `dashboard/app/globals.css`:

```css
:root {
  --primary: 217.2 91.2% 59.8%;  /* Blue */
  --secondary: 217.2 32.6% 17.5%;
  --accent: 217.2 32.6% 17.5%;
}
```

### Animation Speed

Edit `tailwind.config.js`:

```js
animation: {
  "fade-in": "fade-in 0.5s ease-out",  // Change duration here
}
```

### Add Custom Components

```powershell
# Use shadcn/ui CLI to add components
npx shadcn-ui@latest add button
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add tabs
```

---

## 🚀 Deployment

### Vercel (Recommended)

```powershell
# Install Vercel CLI
npm install -g vercel

# Deploy
cd dashboard
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

```powershell
docker build -t fraud-dashboard .
docker run -p 3000:3000 fraud-dashboard
```

---

## 📊 Example API Endpoints Needed

Your ML backend should expose these endpoints:

```python
# api/main.py (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/metrics")
async def get_metrics():
    return {
        "total_transactions": 247891,
        "fraud_detected": 1042,
        "model_accuracy": 95.8,
        "active_users": 9841,
    }

@app.post("/api/predict")
async def predict_fraud(transaction_id: str):
    # Load TGN model and predict
    prediction = model.predict(transaction_id)
    return {
        "transaction_id": transaction_id,
        "fraud_probability": 0.92,
        "risk_level": "high",
    }

@app.get("/api/recent-detections")
async def get_recent_detections():
    return [
        {
            "id": "TXN-8421",
            "timestamp": "2024-01-15T10:30:00Z",
            "amount": 12450,
            "risk": "high",
            "status": "blocked",
        }
    ]
```

---

## 🐛 Troubleshooting

### Issue: Module not found errors

**Solution:** Install dependencies
```powershell
cd dashboard
npm install
```

### Issue: Tailwind styles not loading

**Solution:** Ensure `globals.css` is imported in `app/layout.tsx`

### Issue: Framer Motion animations not working

**Solution:** Make sure components use `'use client'` directive

### Issue: CORS errors when connecting to backend

**Solution:** Add CORS middleware to FastAPI (see example above)

---

## 🎯 Next Steps

1. **Install dependencies:**
   ```powershell
   cd dashboard
   npm install
   ```

2. **Start development server:**
   ```powershell
   npm run dev
   ```

3. **Create ML backend API** (see `api/` folder)

4. **Connect dashboard to backend** (edit `lib/api.ts`)

5. **Customize theme and animations** (edit `globals.css`)

6. **Deploy to Vercel** when ready

---

## 📚 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Framer Motion](https://www.framer.com/motion/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [shadcn/ui](https://ui.shadcn.com/)
- [Recharts](https://recharts.org/)
- [Three.js](https://threejs.org/docs/)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)

---

## 🤝 Contributing

This dashboard is part of the **DSL501 Financial Fraud Detection** project. See main `README.md` for project overview.

**Built with ❤️ for industrial-scale fraud detection**
