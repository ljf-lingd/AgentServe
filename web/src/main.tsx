import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Alert, Button, Card, ConfigProvider, Form, Input, Segmented, Space, Spin, Tag, Typography } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { XProvider } from '@ant-design/x'
import { errorMessage, post, type Access, type User } from './api'
import { useJson } from './hooks'
import { AdminOverview } from './admin'
import { Workspace } from './workspace'
import 'antd/dist/reset.css'
import './style.css'

function Login({ onLogin, notice }: { onLogin: (user: User) => void; notice?: string }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  async function login(credentials: { username: string; password: string }) {
    setBusy(true)
    setError(undefined)
    try {
      onLogin(await post<User>('/auth/login', credentials))
    } catch (error) {
      setError(errorMessage(error))
    } finally {
      setBusy(false)
    }
  }
  return <main className="login">
    <Card title="登录 AgentServe">
      {(error || notice) && <Alert type={error ? 'error' : 'info'} title={error ?? notice} />}
      <Form layout="vertical" onFinish={login} requiredMark={false}>
        <Form.Item name="username" label="账号" rules={[{ required: true }]}>
          <Input autoComplete="username" />
        </Form.Item>
        <Form.Item name="password" label="密码" rules={[{ required: true }]}>
          <Input.Password autoComplete="current-password" />
        </Form.Item>
        <Button block type="primary" htmlType="submit" loading={busy}>登录</Button>
      </Form>
    </Card>
  </main>
}

function App() {
  const { data, error, reload } = useJson<Access>('/auth/me')
  const [override, setOverride] = useState<Access>()
  const [admin, setAdmin] = useState(false)
  const [notice, setNotice] = useState<string>()
  const access = override ?? data

  useEffect(() => {
    function expire() {
      setOverride({ mode: 'auth', user: null })
      setAdmin(false)
      setNotice('登录已过期，请重新登录。')
    }
    window.addEventListener('agentserve-session-expired', expire)
    return () => window.removeEventListener('agentserve-session-expired', expire)
  }, [])

  async function logout() {
    try {
      await post('/auth/logout')
      const prefix = 'agentserve:pending:auth:' + access?.user?.username + ':'
      for (const key of Object.keys(sessionStorage)) {
        if (key.startsWith(prefix)) sessionStorage.removeItem(key)
      }
      location.hash = 'sql'
      setOverride({ mode: 'auth', user: null })
      setAdmin(false)
      setNotice(undefined)
    } catch (error) {
      setNotice(errorMessage(error))
    }
  }

  if (!access) return <main className="login">{error ?
    <Alert type="error" title={error} action={<Button onClick={reload}>重试</Button>} /> : <Spin />}
  </main>
  const demo = access.mode === 'demo'
  const canManage = demo || access.user?.role === 'admin'
  const identity = demo ? 'demo' : 'auth:' + access.user?.username

  return <>
    <header className="topbar">
      <div className="brand"><strong>AgentServe</strong><span>经营分析助手</span></div>
      <Space wrap>
        {demo && <Tag color="gold">本地演示</Tag>}
        {access.user && <Typography.Text>{access.user.username}</Typography.Text>}
        {canManage && <Segmented value={admin ? '运行概览' : '工作台'}
          options={['工作台', '运行概览']} onChange={(view) => setAdmin(view === '运行概览')} />}
        {access.user && <Button onClick={() => void logout()}>退出</Button>}
      </Space>
    </header>
    {!demo && !access.user ? <Login notice={notice} onLogin={(user) => {
      location.hash = 'sql'
      setNotice(undefined)
      setOverride({ mode: 'auth', user })
    }} /> : <>
      {notice && <Alert type="warning" title={notice} />}
      {admin && canManage ? <AdminOverview /> : <Workspace key={identity} identity={identity} />}
    </>}
  </>
}

createRoot(document.getElementById('root')!).render(
  <ConfigProvider locale={zhCN} theme={{
    token: { colorPrimary: '#176963', borderRadius: 8, colorBgLayout: '#f5f7f8' },
  }}>
    <XProvider><App /></XProvider>
  </ConfigProvider>,
)
