import { expect, test } from '@playwright/test'

test('sign in, reload, and sign out in the V6 trial UI', async ({ page }) => {
  const url = process.env.AGENTSERVE_TEST_AUTH_URL
  const password = process.env.AGENTSERVE_TEST_AUTH_PASSWORD
  test.skip(!url || !password, 'Set the trial API URL and pre-provisioned test password')

  await page.goto(url + '/app/')
  for (const [username, admin] of [['trial-user', false], ['trial-admin', true]] as const) {
    await page.getByLabel('账号').fill(username)
    await page.getByLabel('密码').fill(password!)
    await page.getByRole('button', { name: /^登\s*录$/ }).click()
    await expect(page.getByText(username, { exact: true })).toBeVisible()
    await page.reload()
    await expect(page.getByText(username, { exact: true })).toBeVisible()
    const manage = page.getByText('运行概览', { exact: true })
    if (admin) {
      await manage.click()
      await expect(page.getByText('任务完成耗时（JCT）')).toBeVisible()
    } else {
      await expect(manage).toHaveCount(0)
      await page.getByRole('button', { name: '新建会话' }).click()
      await expect(page.getByPlaceholder('输入经营问题', { exact: false })).toBeVisible()
    }
    await page.getByRole('button', { name: /^退\s*出$/ }).click()
    await expect(page.getByRole('button', { name: /^登\s*录$/ })).toBeVisible()
  }
})
