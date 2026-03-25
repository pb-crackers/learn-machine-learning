# Machine Learning Foundations Course

A self-paced, locally-hosted machine learning course built with Next.js, MDX, and PostgreSQL. Takes you from intermediate Python to understanding transformers and LLMs.

## Quick Start

### Prerequisites
- Node.js 18+
- Docker & Docker Compose

### Setup

```bash
# 1. Start PostgreSQL
docker-compose up -d

# 2. Install dependencies
npm install

# 3. Run database migrations
npx prisma migrate dev --name init

# 4. Seed the database
npm run db:seed

# 5. Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to start learning.

## Course Modules

| Module | Title | Lessons |
|--------|-------|---------|
| 0 | Python for ML | 2 |
| 1 | Math Refreshers | 3 |
| 2 | Classical Machine Learning | 5 |
| 3 | Neural Networks & Deep Learning | 5 |
| 4 | NLP & LLMs | 5 |
| 5 | Capstone Project | 3 |
| **Total** | | **23** |

## Tech Stack

- **Framework:** Next.js 15 (App Router)
- **Content:** MDX with KaTeX math rendering
- **Styling:** Tailwind CSS v4
- **Database:** PostgreSQL + Prisma ORM
- **Code Highlighting:** rehype-pretty-code (Shiki)

## Project Structure

```
├── content/          # MDX lesson files organized by module
├── labs/             # Starter code and datasets per lesson
├── prisma/           # Database schema and migrations
├── src/
│   ├── app/          # Next.js App Router pages and API routes
│   ├── components/   # React components (Quiz, Sidebar, callouts, etc.)
│   └── lib/          # Database client, MDX pipeline, course structure
├── docker-compose.yml
└── package.json
```
