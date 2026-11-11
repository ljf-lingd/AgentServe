# 订单库字段

## 订单状态

orders.status 只接受 completed（已完成）、pending（待处理）和 cancelled（已取消）。

## 客户身份

customers.customer_id 是客户标识。姓名相同的客户可能是不同的人，按客户汇总时使用 customer_id。按城市汇总使用 customers.city。

## 订单与明细

orders.order_id 标识订单。order_items 通过 order_id 关联订单，通过 product_id 关联商品。明细表的联合主键是 (order_id, product_id)。

## 订单日期

订单业务日期取 orders.order_date，使用 YYYY-MM-DD 文本格式。按月统计使用业务日期所属月份。business_orders_v1 是本项目的 2026 年合成验收数据。

## 商品目录字段

products.sku 是商品编码，products.category 是目录分类，products.price_cents 是当前目录标价。城市、分类、姓名和 SKU 的值区分大小写。

## 金额字段

order_items.unit_price_cents 是订单明细的历史成交单价。products.price_cents 是目录当前标价。两个字段都是以整数分记录的金额。
