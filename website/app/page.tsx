'use client'

import { useState, useEffect } from 'react'
import { useTheme } from 'next-themes'
import { Sun, Moon, ChevronRight } from 'lucide-react'
import ReactFlow, { Node, Edge, Background } from 'reactflow'
import 'reactflow/dist/style.css'

const initialNodes: Node[] = [
  { id: '1', position: { x: 250, y: 250 }, data: { label: 'AI Engineering' }, type: 'input' },
  { id: '2', position: { x: 100, y: 100 }, data: { label: 'Prompt Engineering' } },
  { id: '3', position: { x: 400, y: 100 }, data: { label: 'RAG' } },
  { id: '4', position: { x: 100, y: 400 }, data: { label: 'Fine-tuning' } },
  { id: '5', position: { x: 400, y: 400 }, data: { label: 'Deployment' } },
  { id: '6', position: { x: 250, y: 500 }, data: { label: 'AI Agents' } },
]

const initialEdges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', animated: true },
  { id: 'e1-3', source: '1', target: '3', animated: true },
  { id: 'e1-4', source: '1', target: '4', animated: true },
  { id: 'e1-5', source: '1', target: '5', animated: true },
  { id: 'e1-6', source: '1', target: '6', animated: true },
]

interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
}

export default function LandingPage() {
  const [nodes] = useState(initialNodes)
  const [edges] = useState(initialEdges)
  const { theme, setTheme } = useTheme()
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    setTheme('dark')
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [setTheme])

  return (
    <div className="min-h-screen flex flex-col transition-colors duration-300 bg-[#0a0a0a] text-white font-sans">
      <header className={`fixed w-full z-10 transition-all duration-300 ${isScrolled ? 'bg-[#0a0a0a]/80 backdrop-blur-md' : 'bg-transparent'}`}>
        <nav className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#8B5CF6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 18C15.3137 18 18 15.3137 18 12C18 8.68629 15.3137 6 12 6C8.68629 6 6 8.68629 6 12C6 15.3137 8.68629 18 12 18Z" stroke="#8B5CF6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 14C13.1046 14 14 13.1046 14 12C14 10.8954 13.1046 10 12 10C10.8954 10 10 10.8954 10 12C10 13.1046 10.8954 14 12 14Z" fill="#8B5CF6"/>
            </svg>
          </div>
          <div className="flex items-center space-x-6">
            <a href="#roadmaps" className="hover:text-purple-400 transition-colors duration-200">Roadmaps</a>
            <a href="#features" className="hover:text-purple-400 transition-colors duration-200">Features</a>
            <a href="#about" className="hover:text-purple-400 transition-colors duration-200">About</a>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-full bg-gray-800 hover:bg-gray-700 transition-colors duration-200"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </nav>
      </header>

      <main className="flex-grow">
        <section className="h-screen flex flex-col justify-center items-center text-center px-4 bg-grid-pattern">
          <h1 className="text-6xl font-bold mb-2 tracking-tight">
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">AI Engineering</span>
            <span className="block text-4xl mt-2">Academy</span>
          </h1>
          <p className="text-2xl mb-8 text-gray-400">Navigating the World of AI, One Step at a Time</p>
          <div className="flex space-x-4">
            <a href="#roadmaps" className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-full transition duration-300 flex items-center">
              Explore Roadmaps
              <ChevronRight className="ml-2 w-5 h-5" />
            </a>
            <a href="#" className="bg-gray-800 hover:bg-gray-700 text-white font-bold py-2 px-6 rounded-full transition duration-300">
              Join Discord
            </a>
          </div>
        </section>

        <section id="features" className="py-20 px-4 bg-[#0f0f0f]">
          <div className="container mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <FeatureCard
                title="Structured Learning Paths"
                description="Follow our carefully crafted roadmaps to master AI engineering concepts step by step."
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>}
              />
              <FeatureCard
                title="Hands-on Projects"
                description="Apply your knowledge with real-world AI projects and build a compelling portfolio."
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path></svg>}
              />
              <FeatureCard
                title="Expert Mentorship"
                description="Learn from industry professionals and get guidance on your AI engineering journey."
                icon={<svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>}
              />
            </div>
          </div>
        </section>

        <section id="roadmaps" className="py-20 px-4">
          <div className="container mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">Learning Roadmaps</h2>
            <div className="h-[600px] w-full">
              <ReactFlow
                nodes={nodes}
                edges={edges}
                fitView
                className="bg-[#0f0f0f] rounded-lg shadow-lg"
              >
                <Background color="#2D3748" gap={16} />
              </ReactFlow>
            </div>
          </div>
        </section>

        <section id="about" className="py-20 px-4 bg-[#0f0f0f]">
          <div className="container mx-auto">
            <h2 className="text-3xl font-bold mb-8 text-center text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">About AI Engineering Academy</h2>
            <p className="max-w-2xl mx-auto text-center text-gray-400 mb-8">
              AI Engineering Academy is your gateway to mastering the world of Artificial Intelligence. We provide structured learning paths, hands-on projects, and expert mentorship to help you navigate the complex landscape of AI engineering.
            </p>
            <p className="max-w-2xl mx-auto text-center text-gray-400">
              Our mission is to empower the next generation of AI engineers with the skills and knowledge they need to shape the future of technology. Join us on this exciting journey into the world of AI!
            </p>
          </div>
        </section>
      </main>

      <footer className="bg-[#0a0a0a] py-8 border-t border-gray-800">
        <div className="container mx-auto px-4">
          <div className="flex flex-wrap justify-between">
            <div className="w-full md:w-1/3 mb-6 md:mb-0">
              <h3 className="text-lg font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">AI Engineering Academy</h3>
              <p className="text-sm text-gray-400">Empowering the next generation of AI engineers.</p>
            </div>
            <div className="w-full md:w-1/3 mb-6 md:mb-0">
              <h3 className="text-lg font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">Quick Links</h3>
              <ul className="text-sm text-gray-400">
                <li><a href="#" className="hover:text-purple-400 transition-colors duration-200">Home</a></li>
                <li><a href="#roadmaps" className="hover:text-purple-400 transition-colors duration-200">Roadmaps</a></li>
                <li><a href="#features" className="hover:text-purple-400 transition-colors duration-200">Features</a></li>
                <li><a href="#about" className="hover:text-purple-400 transition-colors duration-200">About</a></li>
              </ul>
            </div>
            <div className="w-full md:w-1/3">
              <h3 className="text-lg font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">Connect</h3>
              <ul className="text-sm text-gray-400">
                <li><a href="#" className="hover:text-purple-400 transition-colors duration-200">Twitter</a></li>
                <li><a href="#" className="hover:text-purple-400 transition-colors duration-200">LinkedIn</a></li>
                <li><a href="#" className="hover:text-purple-400 transition-colors duration-200">GitHub</a></li>
                <li><a href="#" className="hover:text-purple-400 transition-colors duration-200">Discord</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-8 text-center text-sm text-gray-400">
            <p>&copy; {new Date().getFullYear()} AI Engineering Academy. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

const FeatureCard: React.FC<FeatureCardProps> = ({ title, description, icon }) => {
  return (
    <div className="bg-[#1a1a1a] p-6 rounded-lg shadow-lg">
      <div className="text-purple-500 mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  );
};

